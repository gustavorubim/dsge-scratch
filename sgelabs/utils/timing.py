from __future__ import annotations

import re
from typing import Iterable, List, Tuple

TIME_INDEX_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([+-]?\d+)\s*\)")


def alias_for_shift(name: str, shift: int) -> str:
    if shift == 0:
        return name
    if shift > 0:
        return f"{name}_lead{shift}"
    return f"{name}_lag{abs(shift)}"


def parse_alias(symbol_name: str) -> Tuple[str, int]:
    if symbol_name.endswith("_lag0"):
        symbol_name = symbol_name[:-5]
    if symbol_name.endswith("_lead0"):
        symbol_name = symbol_name[:-6]
    if "_lag" in symbol_name:
        base, _, offset = symbol_name.rpartition("_lag")
        return base, -int(offset)
    if "_lead" in symbol_name:
        base, _, offset = symbol_name.rpartition("_lead")
        return base, int(offset)
    return symbol_name, 0


def replace_time_indices(expr: str, allowed: Iterable[str]) -> str:
    allowed_set = set(allowed)

    def repl(match: re.Match[str]) -> str:
        var, shift = match.group(1), int(match.group(2))
        if var not in allowed_set:
            return match.group(0)
        return alias_for_shift(var, shift)

    return TIME_INDEX_RE.sub(repl, expr)


def collect_shifts(expressions: Iterable[str], names: Iterable[str]) -> dict[str, List[int]]:
    name_set = set(names)
    shifts: dict[str, set[int]] = {name: {0} for name in name_set}
    for expr in expressions:
        for var, shift_str in TIME_INDEX_RE.findall(expr):
            if var in name_set:
                shifts[var].add(int(shift_str))
    return {name: sorted(values) for name, values in shifts.items()}


__all__ = [
    "TIME_INDEX_RE",
    "alias_for_shift",
    "parse_alias",
    "replace_time_indices",
    "collect_shifts",
]
