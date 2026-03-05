"""Prototype modules for YOLO26-Nova research.

These blocks are designed for experimentation and paper ablation.
They are intentionally compact and framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConvBNAct(nn.Module):
    def __init__(self, c: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__()
        p = ((k - 1) // 2) * d
        self.conv = nn.Conv2d(c, c, k, s, p, dilation=d, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SREBlock(nn.Module):
    """Sparse Re-parameterized Elastic Block.

    Train-time: multi-branch with soft gates.
    Deploy-time: can be converted to a single branch by selecting strongest gate.
    """

    def __init__(self, c: int, hidden_ratio: float = 1.0, tau: float = 1.0):
        super().__init__()
        h = int(c * hidden_ratio)
        self.tau = tau

        self.b1 = nn.Sequential(DWConvBNAct(c, 3), ConvBNAct(c, h, 1), ConvBNAct(h, c, 1, act=False))
        self.b2 = nn.Sequential(DWConvBNAct(c, 5), ConvBNAct(c, h, 1), ConvBNAct(h, c, 1, act=False))
        self.b3 = ConvBNAct(c, c, 1, act=False)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(c // 4, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c // 4, 8), 3, 1),
        )
        self.act = nn.SiLU(inplace=True)

        self.deploy = False
        self.deploy_branch_idx = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            branch = [self.b1, self.b2, self.b3][self.deploy_branch_idx]
            return self.act(x + branch(x))

        logits = self.gate(x).flatten(1)
        g = F.softmax(logits / self.tau, dim=1)

        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y = g[:, 0, None, None, None] * y1 + g[:, 1, None, None, None] * y2 + g[:, 2, None, None, None] * y3
        return self.act(x + y)

    @torch.no_grad()
    def switch_to_deploy(self, sample: torch.Tensor) -> int:
        """Select dominant branch from a calibration sample and freeze it."""
        logits = self.gate(sample).flatten(1)
        mean_g = F.softmax(logits / self.tau, dim=1).mean(0)
        self.deploy_branch_idx = int(torch.argmax(mean_g).item())
        self.deploy = True
        return self.deploy_branch_idx


class C3SREB(nn.Module):
    """C3-like block where bottlenecks are replaced by SREBlock."""

    def __init__(self, c1: int, c2: int, n: int = 2, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 1)
        self.cv2 = ConvBNAct(c1, c_, 1)
        self.blocks = nn.Sequential(*[SREBlock(c_) for _ in range(n)])
        self.cv3 = ConvBNAct(2 * c_, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.blocks(self.cv1(x)), self.cv2(x)], dim=1))


class EdgeGate(nn.Module):
    """Hard-concrete-like gate (simplified straight-through sigmoid gate)."""

    def __init__(self, init: float = 0.0, temperature: float = 2.0):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor([init], dtype=torch.float32))
        self.temperature = temperature

    def forward(self, training: bool = True) -> torch.Tensor:
        if training:
            u = torch.rand_like(self.logit).clamp_(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.logit) / self.temperature)
        else:
            s = torch.sigmoid(self.logit)
        hard = (s > 0.5).float()
        return hard.detach() - s.detach() + s


class SparseRouterNeck(nn.Module):
    """Sparse cross-scale router for 3 feature levels (P3, P4, P5)."""

    def __init__(self, channels: Tuple[int, int, int]):
        super().__init__()
        c3, c4, c5 = channels
        self.p3_to_p4 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), ConvBNAct(c4, c4, 1))
        self.p4_to_p3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), ConvBNAct(c4, c3, 1))

        self.p4_to_p5 = nn.Sequential(ConvBNAct(c4, c5, 3, 2), ConvBNAct(c5, c5, 1))
        self.p5_to_p4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), ConvBNAct(c5, c4, 1))

        self.g34 = EdgeGate()
        self.g43 = EdgeGate()
        self.g45 = EdgeGate()
        self.g54 = EdgeGate()

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z34 = self.g34(self.training)
        z43 = self.g43(self.training)
        z45 = self.g45(self.training)
        z54 = self.g54(self.training)

        p4 = p4 + z34 * self.p3_to_p4(p3) + z54 * self.p5_to_p4(p5)
        p3 = p3 + z43 * self.p4_to_p3(p4)
        p5 = p5 + z45 * self.p4_to_p5(p4)

        sparse_penalty = z34 + z43 + z45 + z54
        return p3, p4, p5, sparse_penalty


@dataclass
class UDASTerms:
    loss: torch.Tensor
    det_o2o: torch.Tensor
    det_o2m: torch.Tensor
    distill: torch.Tensor
    sparse: torch.Tensor


class UDASLoss(nn.Module):
    """Uncertainty-aware Dual Assignment Schedule loss wrapper.

    Inputs are already-computed scalar losses for easier integration in existing trainers.
    """

    def __init__(self, alpha_start: float = 0.8, alpha_end: float = 0.1, beta_max: float = 0.5, total_steps: int = 100000):
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_max = beta_max
        self.total_steps = max(total_steps, 1)

    def _alpha(self, step: int) -> float:
        t = min(max(step / self.total_steps, 0.0), 1.0)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * t

    def _beta(self, step: int) -> float:
        t = min(max(step / self.total_steps, 0.0), 1.0)
        return self.beta_max * (1.0 - torch.cos(torch.tensor(t * 3.1415926535))).item() * 0.5

    def forward(
        self,
        det_o2o: torch.Tensor,
        det_o2m: torch.Tensor,
        distill_kl: torch.Tensor,
        sparse_penalty: torch.Tensor,
        step: int,
        lambda_sparse: float = 1e-4,
    ) -> UDASTerms:
        a = self._alpha(step)
        b = self._beta(step)
        loss = det_o2o + a * det_o2m + b * distill_kl + lambda_sparse * sparse_penalty
        return UDASTerms(loss=loss, det_o2o=det_o2o, det_o2m=det_o2m, distill=distill_kl, sparse=sparse_penalty)

