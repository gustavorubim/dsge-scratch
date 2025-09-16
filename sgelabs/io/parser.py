from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import sympy as sp

from sgelabs.ir.model import Equation, ModelIR, Shock, Variable
from sgelabs.utils.timing import alias_for_shift, collect_shifts, replace_time_indices

_ASSIGNMENT_TARGET_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _strip_comments(text: str) -> str:
    clean_lines: List[str] = []
    inside_block = False
    block_buffer: List[str] = []
    i = 0
    while i < len(text):
        if not inside_block and text.startswith("/*", i):
            inside_block = True
            i += 2
            continue
        if inside_block and text.startswith("*/", i):
            inside_block = False
            i += 2
            continue
        if inside_block:
            i += 1
            continue
        block_buffer.append(text[i])
        i += 1
    stripped_block = "".join(block_buffer)
    for raw in stripped_block.splitlines():
        line = raw
        for marker in ("//", "%", "#"):
            idx = line.find(marker)
            if idx != -1:
                line = line[:idx]
        clean_lines.append(line)
    return "\n".join(clean_lines)


def _parse_name_list(statement: str, keyword: str) -> List[str]:
    body = statement[len(keyword) :].strip()
    if not body.endswith(";"):
        raise ValueError(f"Expected ';' terminating {keyword} statement: {statement!r}")
    body = body[:-1].strip()
    if not body:
        return []
    return [token.strip() for token in body.replace(",", " ").split() if token.strip()]


def parse_mod_file(path: str | Path) -> ModelIR:
    mod_path = Path(path)
    text = _strip_comments(mod_path.read_text(encoding="utf-8"))

    endo: List[Variable] = []
    exo: List[Shock] = []
    param_names: List[str] = []
    params: Dict[str, float] = {}
    param_exprs: Dict[str, sp.Expr] = {}
    raw_equations: List[str] = []
    initvals: Dict[str, float] = {}
    shocks: Dict[str, float] = {}
    varobs: List[str] = []

    state: str | None = None
    buffer: List[str] = []
    current_shock: str | None = None

    def flush_declaration(keyword: str) -> None:
        nonlocal buffer
        statement = " ".join(buffer)
        buffer = []
        names = _parse_name_list(statement, keyword)
        if keyword == "var":
            endo.extend(Variable(name) for name in names)
        elif keyword == "varexo":
            exo.extend(Shock(name) for name in names)
        elif keyword == "parameters":
            param_names.extend(names)
        else:  # pragma: no cover - guarded usage
            raise ValueError(f"Unknown declaration keyword {keyword}")

    param_symbols: Dict[str, sp.Symbol] = {}

    lines = [line.strip() for line in text.splitlines()]
    for raw_line in lines:
        if not raw_line:
            continue
        lower = raw_line.lower()
        if state is None:
            if lower.startswith("var "):
                buffer = [raw_line]
                if raw_line.endswith(";"):
                    flush_declaration("var")
                else:
                    state = "var"
            elif lower.startswith("varexo "):
                buffer = [raw_line]
                if raw_line.endswith(";"):
                    flush_declaration("varexo")
                else:
                    state = "varexo"
            elif lower.startswith("parameters "):
                buffer = [raw_line]
                if raw_line.endswith(";"):
                    flush_declaration("parameters")
                    param_symbols = {name: sp.symbols(name) for name in param_names}
                else:
                    state = "parameters"
            elif lower.startswith("model"):
                state = "model"
                buffer = []
            elif lower == "initval;":
                state = "initval"
            elif lower == "shocks;":
                state = "shocks"
                current_shock = None
            elif lower.startswith("varobs"):
                names = _parse_name_list(raw_line, "varobs")
                varobs.extend(names)
            elif "=" in raw_line and raw_line.endswith(";"):
                name, expr = raw_line[:-1].split("=", 1)
                target = name.strip()
                if not _ASSIGNMENT_TARGET_RE.fullmatch(target):
                    continue
                expr_sym = sp.sympify(
                    expr.strip(), locals=params | initvals | param_symbols
                )
                if expr_sym.free_symbols:
                    param_exprs[target] = expr_sym
                else:
                    params[target] = float(sp.N(expr_sym))
            else:
                continue
        elif state in {"var", "varexo", "parameters"}:
            buffer.append(raw_line)
            if raw_line.endswith(";"):
                flush_declaration(state)
                if state == "parameters":
                    param_symbols = {name: sp.symbols(name) for name in param_names}
                state = None
        elif state == "model":
            if lower == "end;":
                state = None
                buffer = []
                continue
            buffer.append(raw_line)
            if raw_line.endswith(";"):
                statement = " ".join(buffer)
                buffer = []
                stmt_clean = statement[:-1].strip()
                if stmt_clean:
                    raw_equations.append(stmt_clean)
        elif state == "initval":
            if lower == "end;":
                state = None
                continue
            if not raw_line.endswith(";"):
                raise ValueError(f"Expected ';' in initval line: {raw_line}")
            assignment = raw_line[:-1]
            if "=" not in assignment:
                raise ValueError(f"Expected '=' in initval assignment: {raw_line}")
            name, expr = assignment.split("=", 1)
            key = name.strip()
            expr_sym = sp.sympify(expr.strip(), locals=params | initvals | param_symbols)
            if expr_sym.free_symbols:
                raise ValueError(f"Initval for {key} depends on symbols: {expr}")
            initvals[key] = float(sp.N(expr_sym))
        elif state == "shocks":
            if lower == "end;":
                state = None
                current_shock = None
                continue
            if lower.startswith("var ") and raw_line.endswith(";"):
                shock_name = raw_line[4:-1].strip()
                current_shock = shock_name
            elif lower.startswith("stderr") and raw_line.endswith(";"):
                if current_shock is None:
                    raise ValueError("stderr specified before shock variable")
                expr = raw_line[len("stderr") : -1].strip()
                expr_sym = sp.sympify(expr, locals=params | param_symbols)
                if expr_sym.free_symbols:
                    raise ValueError(
                        f"Shock stderr for {current_shock} depends on symbols: {expr}"
                    )
                shocks[current_shock] = float(sp.N(expr_sym))
                current_shock = None
        else:
            raise ValueError(f"Unhandled parser state {state}")

    unresolved = True
    while param_exprs and unresolved:
        unresolved = False
        for name in list(param_exprs.keys()):
            expr = param_exprs[name]
            expr_val = sp.N(expr.subs({sp.Symbol(k): v for k, v in params.items()}))
            if expr_val.free_symbols:
                continue
            params[name] = float(expr_val)
            del param_exprs[name]
            unresolved = True
    if param_exprs:
        missing = ", ".join(sorted(param_exprs))
        raise ValueError(f"Unable to resolve parameters: {missing}")

    endo_names = [var.name for var in endo]
    all_names = endo_names + [shock.name for shock in exo]
    shifts = collect_shifts(raw_equations, all_names)

    symbol_map: Dict[str, sp.Symbol] = {}
    for name in all_names:
        for shift in shifts.get(name, [0]):
            alias = alias_for_shift(name, shift)
            symbol_map[alias] = sp.symbols(alias)

    param_symbol_map: Dict[str, sp.Symbol] = {name: sp.symbols(name) for name in param_names}

    equations: List[Equation] = []
    sympy_locals = symbol_map | param_symbol_map | params
    for eq_text in raw_equations:
        normalized = replace_time_indices(eq_text, all_names)
        if "=" not in normalized:
            raise ValueError(f"Model equation missing '=': {eq_text}")
        lhs_text, rhs_text = normalized.split("=", 1)
        lhs_expr = sp.sympify(lhs_text.strip(), locals=sympy_locals)
        rhs_expr = sp.sympify(rhs_text.strip(), locals=sympy_locals)
        equations.append(Equation(lhs=lhs_expr, rhs=rhs_expr))

    for var in endo_names:
        initvals.setdefault(var, 0.0)

    for shock in exo:
        shocks.setdefault(shock.name, 1.0)

    return ModelIR(
        endo=endo,
        exo=exo,
        params=params,
        equations=equations,
        shocks=shocks,
        initvals=initvals,
        varobs=varobs,
    )


__all__ = ["parse_mod_file"]
