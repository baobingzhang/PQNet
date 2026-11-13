"""Tests encode the design claims of the paper, not just shapes."""

import numpy as np
import pytest
import torch

from pqnet import PQNet, discover_periods
from pqnet.data import _borders

L, H, C = 96, 24, 5


def _pair(periods=(24,), revin=True):
    torch.manual_seed(0)
    pq = PQNet(L, H, C, list(periods), d_model=32, use_revin=revin, residual=True).eval()
    torch.manual_seed(0)
    tq = PQNet(L, H, C, list(periods), d_model=32, use_revin=revin, residual=False).eval()
    return pq, tq


def test_tqnet_is_the_special_case_without_residual():
    """R is initialised to zero, so an untrained PQNet equals TQNet with the same weights."""
    pq, tq = _pair()
    tq.load_state_dict({k: v for k, v in pq.state_dict().items() if not k.startswith("R.")})
    x, t = torch.randn(4, L, C), torch.randint(1000, 9000, (4,))
    assert torch.allclose(pq(x, t), tq(x, t), atol=1e-6)


def test_output_depends_only_on_phase():
    """Shifting absolute time by a full period must not change the forecast."""
    pq, _ = _pair(periods=(24, 168))
    for p in (pq.Q.P[0], pq.Q.P[1], pq.R.P[0], pq.R.P[1]):
        p.data.normal_()
    x, t = torch.randn(3, L, C), torch.randint(1000, 9000, (3,))
    assert torch.allclose(pq(x, t), pq(x, t + 168), atol=1e-5)
    assert not torch.allclose(pq(x, t), pq(x, t + 1), atol=1e-3)


def test_residual_is_read_at_future_phases():
    """Forecast step h receives row (t_end + h) mod P of R (checked without RevIN)."""
    pq, tq = _pair(revin=False)
    pq.R.P[0].data.normal_()
    tq.load_state_dict({k: v for k, v in pq.state_dict().items() if not k.startswith("R.")})
    x, t = torch.randn(2, L, C), torch.tensor([1000, 1013])
    diff = pq(x, t) - tq(x, t)
    for b in range(2):
        rows = (t[b] + torch.arange(H)) % 24
        assert torch.allclose(diff[b], pq.R.P[0].data[rows], atol=1e-5)


def _series(T, period=24, trend=0.0, weekly=False, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    x = np.sin(2 * np.pi * t / period) + trend * t / T + 0.2 * rng.normal(size=(T, 3))
    if weekly:
        x = x + 0.8 * (((t // 24) % 7) < 5) * np.clip(np.sin(2 * np.pi * (t % 24) / 24), 0, None)
    return x


def test_discovery_finds_daily_period_despite_trend():
    assert discover_periods(_series(5000, trend=5.0))[0] == [24]


def test_discovery_abstains_on_random_walk():
    rw = np.cumsum(np.random.default_rng(1).normal(size=(5000, 3)), 0)
    assert discover_periods(rw)[0] == [1]


def test_discovery_proposes_daily_then_weekly_hierarchy():
    assert discover_periods(_series(24 * 7 * 40, weekly=True))[0] == [24, 168]


def test_capacity_bound_excludes_weekly_cycle_on_short_series():
    """With fewer than rho=10 weekly repetitions the weekly lag must not be proposed."""
    assert 168 not in discover_periods(_series(24 * 7 * 8, weekly=True))[0]


@pytest.mark.parametrize("reader,n,expect", [
    ("ett_hour", 17420, ([0, 8544, 11424], [8640, 11520, 14400])),
    ("custom", 1000, ([0, 604, 704], [700, 800, 1000])),
    ("pems", 1000, ([0, 504, 704], [600, 800, 1000])),
])
def test_splits_match_tqnet(reader, n, expect):
    assert _borders(reader, n, 96) == (expect[0], expect[1])
