"""
CLI entry point for the hyperbolic denoising project.

Usage examples
--------------
  uv run main.py download --dataset cbsd68
  uv run main.py download --dataset bsd
  uv run main.py download --all

  uv run main.py train
  uv run main.py train --sigma 25 --stages 5 --epochs 30
  uv run main.py train --train-on cbsd68
  uv run main.py train --train-on cbsd68 bsd400
  uv run main.py train --loss l1

  uv run main.py evaluate checkpoints/sigma25_final.pt
  uv run main.py evaluate checkpoints/sigma25_final.pt --test-on cbsd68
  uv run main.py evaluate checkpoints/sigma25_final.pt --show-diffusion
"""

import typer
from typing import Optional, List
from rich.console import Console

app = typer.Typer(
    name="denoise",
    help="Hyperbolic PDE denoising model — train, evaluate, download data.",
    add_completion=False,
)
console = Console(record=True)

AVAILABLE_TRAIN_DATASETS = ["foe400", "cbsd68", "bsd400"]
AVAILABLE_TEST_DATASETS  = ["cbsd68", "bsd68"]


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

@app.command()
def download(
    dataset: Optional[str] = typer.Option(
        None, "--dataset", "-d",
        help=f"Dataset to download: {AVAILABLE_TRAIN_DATASETS + ['bsd']}. Use 'foe' for FoETrainingSets176 (requires zip in data/).",
    ),
    all_datasets: bool = typer.Option(False, "--all", help="Download all datasets."),
    data_root: str = typer.Option("data", "--data-root"),
):
    """Download dataset(s) to disk."""
    from src.params import DEFAULT_CONFIG
    from src.data import download_bsds500, download_cbsd68

    cfg = DEFAULT_CONFIG
    cfg.data.data_root = data_root

    targets = []
    if all_datasets:
        targets = ["bsd", "cbsd68", "foe"]
    elif dataset:
        targets = [dataset.lower()]
    else:
        console.print("[red]Specify --dataset or --all.[/red]")
        raise typer.Exit(1)

    for t in targets:
        if t == "bsd":
            download_bsds500(cfg.data.data_root, cfg.data.bsds500_url)
        elif t == "cbsd68":
            download_cbsd68(cfg.data.data_root, cfg.data.cbsd68_hf_name)
        elif t == "foe":
            from src.data import download_foe400
            download_foe400(cfg.data.data_root)
        else:
            console.print(f"[red]Unknown dataset '{t}'. Choose: bsd, cbsd68, foe[/red]")
            raise typer.Exit(1)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    sigma: int = typer.Option(25,   "--sigma",       help="Noise level sigma."),
    stages: int = typer.Option(5,   "--stages",      help="Number of unrolled stages T (NOT epochs)."),
    filters: int = typer.Option(24, "--filters",     help="Number of filters K per stage."),
    filter_size: int = typer.Option(7, "--filter-size", help="Filter spatial size."),
    epochs: int = typer.Option(30,  "--epochs",      help="Epochs per stage (one epoch = one full pass over training data)."),
    batch_size: int = typer.Option(128, "--batch-size"),
    lr_filters: float = typer.Option(1e-3, "--lr-filters"),
    lr_scalars: float = typer.Option(1e-2, "--lr-scalars"),
    loss: str = typer.Option("mse", "--loss",        help="Loss function: mse | l1  (mse matches TNRD paper)"),
    phi: str = typer.Option(
        "soft_threshold", "--phi",
        help="Influence fn (fixed): soft_threshold | gaussian_deriv | lorentzian",
    ),
    gamma: float = typer.Option(0.5, "--gamma",      help="Damping coefficient gamma (fixed, not learned)."),
    train_on: Optional[List[str]] = typer.Option(
        None, "--train-on",
        help=f"Training dataset(s). Choices: {AVAILABLE_TRAIN_DATASETS}. Default: foe400.",
    ),
    device: str = typer.Option("auto", "--device",  help="auto | cuda | mps | cpu"),
    data_root: str = typer.Option("data", "--data-root"),
    checkpoint_dir: str = typer.Option("checkpoints", "--checkpoint-dir"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use synthetic data if no real images (debug only)."),
    visualize_every: int = typer.Option(5, "--visualize-every",
        help="Save a training snapshot image every N epochs per stage (0=disable). Shows diffusion progress."),
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate", help="Run full evaluation after training."),
    test_on: Optional[List[str]] = typer.Option(
        None, "--test-on",
        help=f"Test dataset(s) for final evaluation. Choices: {AVAILABLE_TEST_DATASETS}.",
    ),
    show_diffusion: bool = typer.Option(
        True, "--show-diffusion/--no-diffusion",
        help="Save per-stage diffusion grids for final evaluation.",
    ),
    save_images: bool = typer.Option(
        True, "--save-images/--no-save-images",
        help="Save denoised output images for final evaluation.",
    ),
):
    """Train the hyperbolic denoising model (greedy stage-wise)."""
    from src.params import Config, ModelParams, TrainParams, DataParams, EvalParams
    from src.train import train as run_train, evaluate as run_eval
    from src.utils import get_device
    from typing import Literal

    _train_datasets = train_on if train_on else ["foe400"]

    # Validate
    for d in _train_datasets:
        if d not in AVAILABLE_TRAIN_DATASETS:
            console.print(f"[red]Unknown train dataset '{d}'. Choose from: {AVAILABLE_TRAIN_DATASETS}[/red]")
            raise typer.Exit(1)
    if loss not in ("mse", "l1"):
        console.print(f"[red]Unknown loss '{loss}'. Choose: mse | l1[/red]")
        raise typer.Exit(1)

    from pathlib import Path
    from datetime import datetime
    import json

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        model=ModelParams(
            K=filters,
            filter_size=filter_size,
            T=stages,
            gamma=gamma,
            phi=phi,  # type: ignore
        ),
        train=TrainParams(
            output_root="runs",
            sigma=sigma,
            epochs_per_stage=epochs,
            batch_size=batch_size,
            lr_filters=lr_filters,
            lr_scalars=lr_scalars,
            loss_fn=loss,  # type: ignore
            device=device,
            checkpoint_dir=str(run_dir / "checkpoints"),
            visualize_every=visualize_every,
        ),
        data=DataParams(
            data_root=data_root,
            train_datasets=_train_datasets,
            test_datasets=["cbsd68"],   # default
            allow_synthetic=synthetic,
        ),
        eval=EvalParams(
            output_dir=str(run_dir / "outputs"),
            save_images=save_images,
            show_diffusion_stages=show_diffusion
        ),
        run_dir=str(run_dir)
    )

    # Log experimental setup
    with open(run_dir / "config.json", "w") as f:
        # Simple hack to serialize dataclasses
        setup = {
            "model": cfg.model.__dict__,
            "train": cfg.train.__dict__,
            "data": cfg.data.__dict__,
            "eval": cfg.eval.__dict__
        }
        json.dump(setup, f, indent=4)

    console.print(f"\n[bold green]Run directory:[/bold green] {run_dir}")
    console.print(f"[bold]Training datasets :[/bold] {_train_datasets}")
    console.print(f"[bold]Test dataset      :[/bold] cbsd68")
    console.print(f"[bold]Loss function     :[/bold] {loss.upper()}  (MSE matches TNRD paper)")
    console.print(f"[bold]Stages            :[/bold] {stages}  (unrolled PDE steps, NOT epochs)")
    console.print(f"[bold]Epochs per stage  :[/bold] {epochs}  (full passes over training data)\n")

    model, metrics = run_train(cfg)

    # Save training graphs
    from src.plots import save_training_plots
    save_training_plots(metrics, cfg.run_dir)

    if evaluate:
        if test_on:
            for d in test_on:
                if d not in AVAILABLE_TEST_DATASETS:
                    console.print(f"[red]Unknown test dataset '{d}'. Choose from: {AVAILABLE_TEST_DATASETS}[/red]")
                    raise typer.Exit(1)
            cfg.data.test_datasets = test_on

        dev = get_device(device)
        console.print("\n[bold cyan]Starting final evaluation...[/bold cyan]")
        run_eval(model, cfg, device=dev, verbose=True, 
                 save_images=save_images, show_diffusion=show_diffusion)

    # Save console log
    console.save_text(str(run_dir / "console_output.txt"), clear=False)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@app.command()
def evaluate(
    checkpoint: str = typer.Argument(..., help="Path to checkpoint .pt file."),
    sigma: Optional[int] = typer.Option(None, "--sigma", help="Override noise level."),
    test_on: Optional[List[str]] = typer.Option(
        None, "--test-on",
        help=f"Test dataset(s). Choices: {AVAILABLE_TEST_DATASETS}. Default: all in data/.",
    ),
    show_diffusion: bool = typer.Option(
        True, "--show-diffusion/--no-diffusion",
        help="Save per-stage diffusion grids (noisy | stage1 | ... | clean) to outputs/.",
    ),
    save_images: bool = typer.Option(
        True, "--save-images/--no-save-images",
        help="Save denoised output images to outputs/.",
    ),
    device: str = typer.Option("auto", "--device"),
    data_root: str = typer.Option("data", "--data-root"),
):
    """Evaluate a trained model. Shows per-stage diffusion process visually."""
    from pathlib import Path
    from datetime import datetime
    import torch
    from src.params import Config
    from src.train import load_checkpoint, evaluate as run_eval
    from src.utils import get_device

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("eval_runs") / f"eval_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg: Config = ckpt_data.get("config", Config())
    cfg.train.device = device
    cfg.data.data_root = data_root
    cfg.eval.output_dir = str(run_dir)
    cfg.run_dir = str(run_dir)

    # Resolve test datasets: if not specified, use everything available in data/
    if test_on:
        for d in test_on:
            if d not in AVAILABLE_TEST_DATASETS:
                console.print(f"[red]Unknown test dataset '{d}'. Choose from: {AVAILABLE_TEST_DATASETS}[/red]")
                raise typer.Exit(1)
        cfg.data.test_datasets = test_on
    else:
        # Auto-detect: use whatever exists in data/
        from pathlib import Path
        detected = []
        for d in AVAILABLE_TEST_DATASETS:
            candidate = Path(data_root) / d
            if candidate.exists():
                detected.append(d)
        if detected:
            cfg.data.test_datasets = detected
            console.print(f"[dim]Auto-detected datasets: {detected}[/dim]")
        else:
            cfg.data.test_datasets = AVAILABLE_TEST_DATASETS  # will warn if missing

    cfg.eval.save_images = save_images
    cfg.eval.show_diffusion_stages = show_diffusion

    dev = get_device(device)
    model = load_checkpoint(checkpoint, cfg)

    console.print(f"\n[bold]Evaluating on  :[/bold] {cfg.data.test_datasets}")
    console.print(f"[bold]Show diffusion :[/bold] {show_diffusion}  (outputs/<dataset>/sigma<N>_diffusion/)\n")

    run_eval(model, cfg, device=dev, verbose=True, sigma=sigma,
             save_images=save_images, show_diffusion=show_diffusion)

    # Save console log
    console.save_text(str(run_dir / "console_output.txt"), clear=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
