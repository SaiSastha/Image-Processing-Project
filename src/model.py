"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .params import ModelParams
from .utils import get_phi


class Stage(nn.Module):
    """
    One unrolled stage of the hyperbolic PDE denoiser.

    The RHS spatial diffusion is approximated by a learned filter bank:
        D(u) ≈ sum_k  f_k^T * phi_k(f_k * u)

    The full update:
        u^{n+1} = coeff_curr*u^n + coeff_prev*u^{n-1} + coeff_diff*(D(u^n) - lam*(u^n - u_tilde))
    """

    def __init__(self, params: ModelParams, stage_idx: int = 0):
        super().__init__()
        self.stage_idx = stage_idx
        self.K = params.K
        self.filter_size = params.filter_size
        self.phi = get_phi(params.phi, **_phi_kwargs(params))

        # ---- Learned filters (K, 1, fs, fs) --------------------------------
        # Using 0.02 init scale — safe middle ground for Gaussian learning
        self.filters = nn.Parameter(
            torch.randn(params.K, 1, params.filter_size, params.filter_size) * 0.02
        )

        # ---- Learned scalars ------------------------------------------------
        if params.learn_coeffs:
            self.coeff_curr = nn.Parameter(torch.tensor(params.coeff_curr_init))
            self.coeff_prev = nn.Parameter(torch.tensor(params.coeff_prev_init))
            self.coeff_diff = nn.Parameter(torch.tensor(params.coeff_diff_init))
        else:
            self.register_buffer("coeff_curr", torch.tensor(params.coeff_curr_init))
            self.register_buffer("coeff_prev", torch.tensor(params.coeff_prev_init))
            self.register_buffer("coeff_diff", torch.tensor(params.coeff_diff_init))

        self.lam = nn.Parameter(torch.tensor(params.lambda_init))

    def forward(self, u_curr: Tensor, u_prev: Tensor, u_tilde: Tensor) -> Tensor:
        """
        u_curr  : (B, 1, H, W)  u^n
        u_prev  : (B, 1, H, W)  u^{n-1}
        u_tilde : (B, 1, H, W)  noisy input (fixed anchor for data fidelity)

        Returns u_next : (B, 1, H, W)  u^{n+1}
        """
        pad = self.filter_size // 2

        # 1. Learned spatial diffusion (TNRD filter bank)
        # Zero-mean filters (TNRD stability trick)
        f = self.filters - self.filters.mean(dim=[2, 3], keepdim=True)
        responses = F.conv2d(u_curr, f, padding=pad)
        influenced = self.phi(responses)

        # Adjoint convolution (transpose)
        f_T = f.flip([2, 3])
        diffusion = F.conv2d(
            influenced,
            f_T.permute(1, 0, 2, 3),   # (1, K, fs, fs)
            padding=pad,
        )

        # 2. Data fidelity term
        fidelity = self.lam * (u_curr - u_tilde)

        # 3. Hyperbolic update
        u_next = (
            self.coeff_curr * u_curr
            + self.coeff_prev * u_prev
            + self.coeff_diff * (diffusion - fidelity)
        )

        return u_next


class UnrolledNet(nn.Module):
    """
    Full T-stage unrolled hyperbolic denoising network.
    """

    def __init__(self, params: ModelParams):
        super().__init__()
        self.T = params.T
        self.stages = nn.ModuleList(
            [Stage(params, stage_idx=n) for n in range(params.T)]
        )

    def forward(
        self,
        u_tilde: Tensor,
        train_stage: int | None = None,
    ) -> list[Tensor]:
        """
        Returns list of T outputs, one per stage.
        """
        # Zero initial velocity: u^{-1} = u^0 = u_tilde
        u_prev = u_tilde
        u_curr = u_tilde

        outputs = []
        for n, stage in enumerate(self.stages):
            if train_stage is not None and n != train_stage:
                with torch.no_grad():
                    u_next = stage(u_curr, u_prev, u_tilde)
            else:
                u_next = stage(u_curr, u_prev, u_tilde)

            outputs.append(u_next)
            u_prev = u_curr
            u_curr = u_next

        return outputs

    def final_output(self, u_tilde: Tensor) -> Tensor:
        return self.forward(u_tilde)[-1]

    def output_up_to_stage(self, u_tilde: Tensor, stage_idx: int) -> list[Tensor]:
        return self.forward(u_tilde)[:stage_idx + 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phi_kwargs(params: ModelParams) -> dict:
    if params.phi == "soft_threshold":
        return {"threshold": params.phi_threshold}
    if params.phi == "gaussian_deriv":
        return {"sigma": params.phi_sigma}
    if params.phi == "lorentzian":
        return {"scale": params.phi_scale}
    return {}
