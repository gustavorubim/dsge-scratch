from __future__ import annotations

import re
from typing import Mapping

_LAG_PATTERN = re.compile(r"^(?P<root>.+)_lag(?P<lag>\d+)$")

STATE_OVERRIDES: Mapping[str, str] = {
    "ewma": "Wage Markup Shock (moving average)",
    "epinfma": "Price Markup Shock (moving average)",
}

STATE_BASE_LABELS: Mapping[str, str] = {
    "a": "Technology",
    "b": "Risk Premium",
    "c": "Consumption",
    "ca": "Current Account",
    "g": "Government Spending",
    "gdp": "GDP",
    "h": "Hours Worked",
    "i": "Investment",
    "inve": "Investment",
    "inv": "Investment",
    "k": "Capital Stock",
    "kp": "Installed Capital",
    "l": "Labor",
    "lab": "Labor",
    "m": "Money Stock",
    "mc": "Marginal Cost",
    "ms": "Monetary Policy Shock",
    "n": "Employment",
    "p": "Price Level",
    "pi": "Inflation",
    "pinf": "Inflation",
    "pk": "Tobin's Q",
    "q": "Tobin's Q",
    "qs": "Investment Efficiency",
    "r": "Nominal Interest Rate",
    "R": "Gross Nominal Interest Rate",
    "rn": "Natural Real Rate",
    "rr": "Real Interest Rate",
    "rk": "Rental Rate of Capital",
    "u": "Capacity Utilization",
    "w": "Real Wage",
    "x": "Output Gap",
    "y": "Output",
    "z": "Productivity",
    "zcap": "Capacity Utilization",
    "tfp": "TFP",
    "infl": "Inflation",
    "inflation": "Inflation",
    "spread": "Credit Spread",
    "nx": "Net Exports",
    "tb": "Trade Balance",
    "lambda": "Lagrange Multiplier",
    "lam": "Lagrange Multiplier",
    "mu": "Markup",
    "nu": "Markup",
    "spinf": "Price Markup Shock",
    "sw": "Wage Markup Shock",
    "ew": "Wage Markup Shock",
    "epinf": "Price Markup Shock",
}

SHOCK_OVERRIDES: Mapping[str, str] = {
    "ea": "Technology Shock",
    "eb": "Risk Premium Shock",
    "eg": "Government Spending Shock",
    "em": "Monetary Policy Shock",
    "eqs": "Investment Efficiency Shock",
    "epinf": "Price Markup Shock",
    "ew": "Wage Markup Shock",
    "epinfma": "Price Markup Shock (moving average)",
    "ewma": "Wage Markup Shock (moving average)",
}

SHOCK_BASE_LABELS: Mapping[str, str] = {
    "a": "Technology",
    "b": "Risk Premium",
    "g": "Government Spending",
    "m": "Monetary Policy",
    "pinf": "Price Markup",
    "tfp": "TFP",
    "q": "Output Gap",
    "qs": "Investment Efficiency",
    "r": "Interest Rate",
    "w": "Wage Markup",
    "y": "Output",
}

STATE_SUFFIX_RULES = {
    "_obs": lambda label: f"Observed {label}",
    "obs": lambda label: f"Observed {label}",
    "_gap": lambda label: f"{label} Gap",
    "gap": lambda label: f"{label} Gap",
    "_hat": lambda label: f"{label} (deviation)",
    "hat": lambda label: f"{label} (deviation)",
    "_4": lambda label: f"{label} (four-quarter sum)",
    "4": lambda label: f"{label} (four-quarter sum)",
    "_trend": lambda label: f"{label} Trend",
    "trend": lambda label: f"{label} Trend",
    "_ann": lambda label: f"{label} (annualized)",
    "ann": lambda label: f"{label} (annualized)",
}

STATE_FLEX_SUFFIX = "f"

STATE_PREFIX_RULES = (
    ("d_", "Change in"),
    ("d", "Change in"),
    ("log_", "Log"),
    ("log", "Log"),
    ("ln", "Log"),
)

SHOCK_PREFIXES = ("eps_", "eps", "shock_", "e_", "e")

SHOCK_SUFFIXES = {"_shock", "shock"}


def _lookup(mapping: Mapping[str, str], key: str) -> str | None:
    if key in mapping:
        return mapping[key]
    lowered = key.lower()
    if lowered in mapping:
        return mapping[lowered]
    upper = key.upper()
    if upper in mapping:
        return mapping[upper]
    return None


def _titleize(name: str) -> str:
    cleaned = name.replace("_", " ")
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return name
    titled: list[str] = []
    for token in tokens:
        titled.append(token if token.isupper() else token.capitalize())
    return " ".join(titled)


def _format_label(name: str, overrides: Mapping[str, str], base_labels: Mapping[str, str]) -> str:
    if not name:
        return name

    direct = _lookup(overrides, name)
    if direct:
        return direct

    match = _LAG_PATTERN.match(name)
    if match:
        base = match.group("root")
        lag = match.group("lag")
        base_label = _format_label(base, overrides, base_labels)
        return f"{base_label} (lag {lag})"

    direct = _lookup(base_labels, name)
    if direct:
        return direct

    for suffix, builder in STATE_SUFFIX_RULES.items():
        if name.endswith(suffix):
            root = name[: -len(suffix)]
            if not root:
                break
            base_label = _format_label(root, overrides, base_labels)
            return builder(base_label)

    if name.endswith(STATE_FLEX_SUFFIX) and len(name) > 1:
        base = name[:-1]
        if _lookup(overrides, base) or _lookup(base_labels, base):
            base_label = _format_label(base, overrides, base_labels)
            return f"{base_label} (flexible price)"

    for prefix, description in STATE_PREFIX_RULES:
        if name.startswith(prefix) and len(name) > len(prefix):
            remainder = name[len(prefix) :]
            base_label = _format_label(remainder, overrides, base_labels)
            return f"{description} {base_label}"

    return _titleize(name)


def format_state_label(name: str) -> str:
    return _format_label(name, STATE_OVERRIDES, STATE_BASE_LABELS)


def format_shock_label(name: str) -> str:
    direct = _lookup(SHOCK_OVERRIDES, name)
    if direct:
        return direct

    cleaned = name
    for suffix in SHOCK_SUFFIXES:
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)]
            break

    for prefix in SHOCK_PREFIXES:
        if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
            candidate = cleaned[len(prefix) :]
            direct = _lookup(SHOCK_OVERRIDES, candidate)
            if direct:
                return direct
            cleaned = candidate
            break

    label = _format_label(cleaned, SHOCK_OVERRIDES, SHOCK_BASE_LABELS)
    if "shock" not in label.lower():
        label = f"{label} Shock"
    return label


def _combine_with_original(original: str, label: str) -> str:
    if not label:
        return original
    return f"{original} - {label}"


def format_state_display(name: str) -> str:
    label = format_state_label(name)
    return _combine_with_original(name, label)


def format_shock_display(name: str) -> str:
    label = format_shock_label(name)
    return _combine_with_original(name, label)


__all__ = [
    "format_shock_display",
    "format_shock_label",
    "format_state_display",
    "format_state_label",
]
