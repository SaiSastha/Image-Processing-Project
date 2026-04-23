"""
Greedy (stage-wise) training loop.

Each stage is trained independently:
  1. Freeze all previous stages (no_grad in forward pass).
  2. Optimise only the current stage's parameters.
  3. After training, freeze current stage and move to the next.

Loss: MSE  = (1/2)||u^{(n)} - u_gt||^2   (matches TNRD paper)
Metric: PSNR reported after each stage — evaluation only, NOT the loss.
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .model import UnrolledNet
from .params import Config
from .utils import get_device, psnr
from .data import build_train_loader, build_test_loader

console = Console()


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def _make_criterion(loss_fn: str) -> nn.Module:
    if loss_fn == "mse":
        return nn.MSELoss()
    if loss_fn == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unknown loss_fn '{loss_fn}'. Choose: mse | l1")


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def _make_optimizer(stage: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    filter_params = [stage.filters]
    scalar_params = [p for name, p in stage.named_parameters() if name != "filters"]
    return torch.optim.Adam(
        [
            {"params": filter_params, "lr": cfg.train.lr_filters},
            {"params": scalar_params, "lr": cfg.train.lr_scalars},
        ],
        betas=cfg.train.adam_betas,
        weight_decay=cfg.train.weight_decay,
    )


# ---------------------------------------------------------------------------
# Visual snapshot — saves noisy / staged outputs / clean side-by-side
# ---------------------------------------------------------------------------

def _save_visual_snapshot(
    model: UnrolledNet,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    stage_idx: int,
    epoch: int,
    cfg: Config,
) -> None:
    """
    Saves a side-by-side image: [noisy | stage_0 | ... | stage_n | clean]
    so you can see the diffusion process visually.
    Operates on the first image in the batch only.
    """
    out_dir = Path(cfg.eval.output_dir) / "training_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        sample_noisy = noisy[:1]    # (1,1,H,W)
        sample_clean = clean[:1]
        stage_outputs = model.output_up_to_stage(sample_noisy, stage_idx)

    # Build grid row: noisy, each stage output, clean
    frames = [sample_noisy.clamp(0, 1)] + \
             [o.clamp(0, 1) for o in stage_outputs] + \
             [sample_clean]
    grid = torch.cat(frames, dim=-1)  # concatenate along width

    fname = out_dir / f"stage{stage_idx+1:02d}_epoch{epoch:03d}.png"
    save_image(grid, fname)
    model.train()


# ---------------------------------------------------------------------------
# Single-stage training
# ---------------------------------------------------------------------------

def train_stage(
    model: UnrolledNet,
    stage_idx: int,
    train_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> list[float]:
    """Train one stage greedily. Returns per-epoch loss history."""
    stage = model.stages[stage_idx]
    stage.train()

    optimizer = _make_optimizer(stage, cfg)
    criterion = _make_criterion(cfg.train.loss_fn)
    loss_history = []

    with Progress(
        SpinnerColumn(),
        TextColumn(f"  Stage {stage_idx + 1}/{model.T}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=cfg.train.epochs_per_stage)

        for epoch in range(cfg.train.epochs_per_stage):
            epoch_loss = 0.0
            num_batches = 0
            last_noisy = last_clean = None

            for noisy, clean in train_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                last_noisy, last_clean = noisy, clean

                optimizer.zero_grad()
                outputs = model(noisy, train_stage=stage_idx)
                pred = outputs[stage_idx]

                # MSE loss (TNRD: (1/2)||pred - clean||^2)
                loss = 0.5 * criterion(pred, clean)
                loss.backward()

                if cfg.train.grad_clip is not None:
                    nn.utils.clip_grad_norm_(stage.parameters(), cfg.train.grad_clip)

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append(avg_loss)
            progress.advance(task)

            # Visual snapshot: save image showing diffusion progress
            vis_every = cfg.train.visualize_every
            if vis_every > 0 and (epoch + 1) % vis_every == 0 and last_noisy is not None:
                _save_visual_snapshot(
                    model, last_noisy, last_clean, stage_idx, epoch + 1, cfg
                )

            # Periodic checkpoint
            if (epoch + 1) % cfg.train.save_every == 0:
                _save_checkpoint(model, stage_idx, epoch + 1, cfg)

    return loss_history


# ---------------------------------------------------------------------------
# Full greedy training
# ---------------------------------------------------------------------------

def train(cfg: Config) -> tuple[UnrolledNet, dict]:
    _set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    console.print(f"\n[bold green]Device:[/bold green] {device}")
    
    metrics = {
        "loss": [],  # List of lists (one per stage)
        "psnr": []   # Final PSNR after each stage
    }

    model = UnrolledNet(cfg.model).to(device)
    train_loader = build_train_loader(cfg.data, cfg.train)

    for stage_idx in range(cfg.model.T):
        console.rule(f"[bold]Stage {stage_idx + 1} / {cfg.model.T}")
        losses = train_stage(model, stage_idx, train_loader, cfg, device)
        metrics["loss"].append(losses)

        # Quick eval after each stage — report PSNR for CURRENT stage only
        avg_psnr = evaluate(model, cfg, device, verbose=False, 
                            save_images=False, show_diffusion=False,
                            num_stages=stage_idx + 1)
        metrics["psnr"].append(avg_psnr)

        psnr_str = f" | PSNR: {avg_psnr:.2f} dB" if avg_psnr is not None else ""
        console.print(
            f"  Stage {stage_idx + 1} complete — "
            f"final loss: {losses[-1]:.5f}{psnr_str}\n"
        )

    _save_checkpoint(model, cfg.model.T - 1, cfg.train.epochs_per_stage, cfg, final=True)

    # Save summary results to a file for easy tabulation
    if cfg.run_dir:
        summary_path = Path(cfg.run_dir) / "results_summary.csv"
        with open(summary_path, "w") as f:
            f.write("Stage,Final_Loss,Mean_PSNR_dB\n")
            for i, (loss_list, psnr_val) in enumerate(zip(metrics["loss"], metrics["psnr"])):
                f.write(f"{i+1},{loss_list[-1]:.6f},{psnr_val:.4f}\n")
        console.print(f"[bold green]Results summary saved to:[/bold green] {summary_path}")

    console.print("[bold green]Training complete.[/bold green]")
    return model, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: UnrolledNet,
    cfg: Config,
    device: torch.device | None = None,
    verbose: bool = True,
    sigma: int | None = None,
    save_images: bool | None = None,
    show_diffusion: bool | None = None,
    num_stages: int | None = None,
) -> float | None:
    """
    Evaluate on available test datasets.

    num_stages: if provided, only evaluates using first N stages of the model.
                Useful during greedy training.
    """
    if device is None:
        device = get_device(cfg.train.device)
    sigma = sigma or cfg.train.sigma
    _save = save_images if save_images is not None else cfg.eval.save_images
    _show_diff = show_diffusion if show_diffusion is not None else cfg.eval.show_diffusion_stages
    
    # Target stage for final output
    T_eval = num_stages if num_stages is not None else model.T

    model.eval()
    results = {}

    with torch.no_grad():
        for ds_name in cfg.data.test_datasets:
            loader = build_test_loader(ds_name, cfg.data, sigma)
            if loader is None:
                continue

            psnr_scores = []
            for img_idx, (noisy, clean) in enumerate(loader):
                noisy = noisy.to(device)
                clean = clean.to(device)

                # Get outputs up to target stage
                all_outputs = model.output_up_to_stage(noisy, T_eval - 1)
                pred = all_outputs[-1].clamp(0.0, 1.0)
                psnr_scores.append(psnr(pred, clean))

                # Save final denoised image
                if _save and (cfg.eval.num_save is None or img_idx < cfg.eval.num_save):
                    tag = f"_stage{T_eval}" if num_stages else ""
                    out_dir = Path(cfg.eval.output_dir) / ds_name / f"sigma{sigma}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_image(pred, out_dir / f"{img_idx:04d}_denoised{tag}.png")
                    save_image(noisy.clamp(0, 1), out_dir / f"{img_idx:04d}_noisy.png")
                    save_image(clean.clamp(0, 1), out_dir / f"{img_idx:04d}_clean.png")

                # Save per-stage diffusion grid
                if _show_diff and (cfg.eval.num_save is None or img_idx < cfg.eval.num_save):
                    diff_dir = Path(cfg.eval.output_dir) / ds_name / f"sigma{sigma}_diffusion"
                    diff_dir.mkdir(parents=True, exist_ok=True)
                    frames = (
                        [noisy.clamp(0, 1)]
                        + [o.clamp(0, 1) for o in all_outputs]
                        + [clean.clamp(0, 1)]
                    )
                    grid = torch.cat(frames, dim=-1)
                    save_image(grid, diff_dir / f"{img_idx:04d}_stages.png")

            results[ds_name] = sum(psnr_scores) / len(psnr_scores)

    if verbose and results:
        table = Table(title=f"Evaluation  (σ={sigma}, stages={T_eval})")
        table.add_column("Dataset", style="cyan")
        table.add_column("Mean PSNR (dB)", style="magenta")
        for name, val in results.items():
            table.add_row(name, f"{val:.2f}")
        console.print(table)
    elif verbose:
        console.print("[yellow]⚠️  No test datasets found. Skipping evaluation.[/yellow]")

    model.train()
    return list(results.values())[0] if results else None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: UnrolledNet,
    stage_idx: int,
    epoch: int,
    cfg: Config,
    final: bool = False,
) -> None:
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag = "final" if final else f"stage{stage_idx + 1}_ep{epoch}"
    path = ckpt_dir / f"sigma{cfg.train.sigma}_{tag}.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "stage_idx":   stage_idx,
            "epoch":       epoch,
            "config":      cfg,
        },
        path,
    )
    if final:
        console.print(f"[bold]Checkpoint saved:[/bold] {path}")


def load_checkpoint(path: str, cfg: Config) -> UnrolledNet:
    """
    Load model from checkpoint. 
    """
    device = get_device(cfg.train.device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    saved_cfg: Config = ckpt["config"]
    saved_cfg.train.device = cfg.train.device
    saved_cfg.data.data_root = cfg.data.data_root
    
    model = UnrolledNet(saved_cfg.model).to(device)
    model.load_state_dict(ckpt["model_state"])
    
    console.print(f"Loaded checkpoint: {path} (Stage {ckpt['stage_idx']+1}, Epoch {ckpt['epoch']})")
    return model


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
