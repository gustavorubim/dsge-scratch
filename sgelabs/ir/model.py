from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import sympy as sp


@dataclass(slots=True)
class Variable:
    name: str


@dataclass(slots=True)
class Shock:
    name: str


@dataclass(slots=True)
class Equation:
    lhs: sp.Expr
    rhs: sp.Expr

    def residual(self) -> sp.Expr:
        """Return lhs - rhs expression."""
        return sp.simplify(self.lhs - self.rhs)

    def as_dict(self) -> Dict[str, str]:
        return {"lhs": sp.sstr(self.lhs), "rhs": sp.sstr(self.rhs)}


@dataclass(slots=True)
class ModelIR:
    endo: List[Variable]
    exo: List[Shock]
    params: Dict[str, float]
    equations: List[Equation]
    shocks: Dict[str, float]
    initvals: Dict[str, float]
    varobs: List[str]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "endo": [v.name for v in self.endo],
            "exo": [s.name for s in self.exo],
            "params": dict(self.params),
            "equations": [eq.as_dict() for eq in self.equations],
            "shocks": dict(self.shocks),
            "initvals": dict(self.initvals),
            "varobs": list(self.varobs),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "ModelIR":
        endo = [Variable(name) for name in data["endo"]]
        exo = [Shock(name) for name in data["exo"]]
        params = {str(k): float(v) for k, v in dict(data["params"]).items()}
        equations = [
            Equation(
                lhs=sp.sympify(eq["lhs"], locals=params),
                rhs=sp.sympify(eq["rhs"], locals=params),
            )
            for eq in data["equations"]
        ]
        shocks = {str(k): float(v) for k, v in dict(data["shocks"]).items()}
        initvals = {str(k): float(v) for k, v in dict(data["initvals"]).items()}
        varobs = [str(v) for v in data.get("varobs", [])]
        return cls(
            endo=endo,
            exo=exo,
            params=params,
            equations=equations,
            shocks=shocks,
            initvals=initvals,
            varobs=varobs,
        )


@dataclass(slots=True)
class LinearizedModel:
    gam0: sp.Matrix
    gam1: sp.Matrix
    c: sp.Matrix
    psi: sp.Matrix
    pi: sp.Matrix
    state_names: List[str]
    shock_names: List[str]

    def to_numpy(self) -> Dict[str, "np.ndarray"]:
        import numpy as np

        return {
            "gam0": np.asarray(self.gam0, dtype=float),
            "gam1": np.asarray(self.gam1, dtype=float),
            "c": np.asarray(self.c, dtype=float).reshape(-1, 1),
            "psi": np.asarray(self.psi, dtype=float),
            "pi": np.asarray(self.pi, dtype=float),
        }


__all__ = [
    "Variable",
    "Shock",
    "Equation",
    "ModelIR",
    "LinearizedModel",
]
