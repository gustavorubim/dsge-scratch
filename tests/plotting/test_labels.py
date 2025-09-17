from __future__ import annotations

import pytest

from sgelabs.plotting.labels import (
    format_shock_display,
    format_shock_label,
    format_state_display,
    format_state_label,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("y", "Output"),
        ("pinf_lag2", "Inflation (lag 2)"),
        ("y_obs", "Observed Output"),
        ("labobs", "Observed Labor"),
        ("dy", "Change in Output"),
        ("zcapf", "Capacity Utilization (flexible price)"),
        ("tfp", "TFP"),
        ("unknown_var", "Unknown Var"),
    ],
)
def test_format_state_label(name: str, expected: str) -> None:
    assert format_state_label(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ea", "Technology Shock"),
        ("eps_a", "Technology Shock"),
        ("epinf", "Price Markup Shock"),
        ("epinfma", "Price Markup Shock (moving average)"),
        ("shock_tfp", "TFP Shock"),
        ("em", "Monetary Policy Shock"),
    ],
)
def test_format_shock_label(name: str, expected: str) -> None:
    assert format_shock_label(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("y", "y - Output"),
        ("pinf_lag2", "pinf_lag2 - Inflation (lag 2)"),
        ("zcapf", "zcapf - Capacity Utilization (flexible price)"),
    ],
)
def test_format_state_display(name: str, expected: str) -> None:
    assert format_state_display(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ea", "ea - Technology Shock"),
        ("eps_a", "eps_a - Technology Shock"),
        ("epinfma", "epinfma - Price Markup Shock (moving average)"),
    ],
)
def test_format_shock_display(name: str, expected: str) -> None:
    assert format_shock_display(name) == expected
