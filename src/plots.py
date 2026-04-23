"""
Plotting utilities for training metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def save_training_plots(metrics: dict, output_dir: str) -> str:
    """
    Saves loss and PSNR graphs to the provided run directory.
    """
    out_dir = Path(output_dir) / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss per stage (Combined)
    plt.figure(figsize=(10, 6))
    for i, stage_losses in enumerate(metrics["loss"]):
        plt.plot(stage_losses, label=f"Stage {i+1}")
    plt.title("Greedy Training Loss per Stage")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(out_dir / "loss_stages.png", dpi=300)
    plt.close()

    # 2. PSNR progression across stages
    psnrs = metrics["psnr"]
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(psnrs) + 1), psnrs, marker='o', linestyle='-', color='magenta')
    plt.title("Mean PSNR Progression (Evaluation)")
    plt.xlabel("Total Stages")
    plt.ylabel("PSNR (dB)")
    plt.xticks(range(1, len(psnrs) + 1))
    plt.grid(True, alpha=0.3)
    
    # Annotate points
    for i, val in enumerate(psnrs):
        plt.annotate(f"{val:.2f}", (i + 1, val), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.savefig(out_dir / "psnr_progression.png", dpi=300)
    plt.close()

    print(f"📈 Paper-ready graphs saved to: {out_dir}/")
    return str(out_dir)
