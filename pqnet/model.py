"""PQNet: a phase-indexed periodic memory used on both sides of the forecast.

Time indexing: ``t_end`` is the absolute index (in the full series) of the first
forecast step. Input steps are t_end-L .. t_end-1 and forecast steps are
t_end .. t_end+H-1. The phase of step tau with respect to period P is tau mod P.

With ``residual=False`` and ``periods=[W]`` the model is TQNet (Lin et al., ICML 2025):
temporal queries from a learnable W x C table, one channel attention layer and a
shared two-layer MLP.
"""

import torch
import torch.nn as nn


class PhaseMemory(nn.Module):
    """Sum over periods P of a learnable P x C table read at phase tau mod P."""

    def __init__(self, periods, n_channels):
        super().__init__()
        self.periods = [int(p) for p in periods]
        self.P = nn.ParameterList(
            [nn.Parameter(torch.zeros(p, n_channels)) for p in self.periods]
        )

    def forward(self, t0, length):
        """Read the memory at absolute steps t0 .. t0+length-1. t0: (B,) -> (B, length, C)."""
        t = t0.view(-1, 1) + torch.arange(length, device=t0.device).view(1, -1)
        out = 0
        for p, table in zip(self.periods, self.P):
            out = out + table[torch.remainder(t, p)]
        return out


class PQNet(nn.Module):
    """Query memory Q read at the input phases + residual memory R read at the future phases.

    Args:
        seq_len: look-back length L.
        pred_len: horizon H.
        enc_in: number of channels C.
        periods: list of periods P (one table per period, shared by Q and R).
        d_model: width of the shared MLP.
        dropout: dropout before the output projection.
        use_revin: reversible instance normalisation of every input window.
        residual: add the future-phase residual memory R (False gives TQNet).
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        periods,
        d_model=512,
        dropout=0.0,
        use_revin=True,
        residual=True,
    ):
        super().__init__()
        self.L, self.H, self.C = seq_len, pred_len, enc_in
        self.periods = [int(p) for p in periods]
        self.use_revin, self.residual = use_revin, residual

        # The construction order fixes the parameter initialisation for a given seed.
        self.Q = PhaseMemory(self.periods, enc_in)
        self.attn = nn.MultiheadAttention(
            embed_dim=seq_len, num_heads=4, batch_first=True, dropout=0.5
        )
        if residual:
            self.R = PhaseMemory(self.periods, enc_in)
        self.input_proj = nn.Linear(seq_len, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, pred_len)
        )

    def forward(self, x, t_end):
        """x: (B, L, C) input window; t_end: (B,) absolute index of the first forecast step."""
        mu = x.mean(1, keepdim=True) if self.use_revin else 0.0
        std = torch.sqrt(x.var(1, keepdim=True) + 1e-5) if self.use_revin else 1.0
        x = (x - mu) / std

        xi = x.permute(0, 2, 1)  # (B, C, L): channels are tokens
        q = self.Q(t_end - self.L, self.L).permute(
            0, 2, 1
        )  # queries at the input phases
        h = self.input_proj(xi + self.attn(q, xi, xi)[0])
        y = self.output_proj(self.mlp(h) + h).permute(0, 2, 1)  # (B, H, C)
        if self.residual:
            y = y + self.R(t_end, self.H)  # periodic baseline at the future phases

        return y * std + mu
