#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.stats import norm  # para Phi(·) y z

# =========================================
# PRNGs mínimos (compatibles con .random())
# =========================================

class LCG:
    # Numerical Recipes (32-bit)
    def __init__(self, seed: int, m: int = 2**32, a: int = 1664525, c: int = 1013904223):
        self.m, self.a, self.c = m, a, c
        self.state = seed % m
    def next_uint(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    def random(self) -> float:
        return self.next_uint() / self.m

class RANDU:
    # X_{n+1} = 65539 * X_n mod 2^31  (históricamente malo, para comparar)
    def __init__(self, seed: int):
        self.m, self.a = 2**31, 65539
        self.state = seed | 1  # forzar impar
    def next_uint(self) -> int:
        self.state = (self.a * self.state) % self.m
        return self.state
    def random(self) -> float:
        return self.next_uint() / self.m

class MiddleSquare:
    # Demostrativo, puede colapsar
    def __init__(self, seed: int, n_digits: int = 8):
        assert n_digits % 2 == 0 and n_digits >= 4
        self.n = n_digits
        self.mod = 10 ** n_digits
        self.state = seed % self.mod or (10**(self.n-1)+1)
    def random(self) -> float:
        y = self.state * self.state
        s = str(y).rjust(2*self.n, '0')
        mid = len(s) // 2
        self.state = int(s[mid - self.n//2 : mid + self.n//2])
        return self.state / self.mod

class BBS:
    # Blum-Blum-Shub DEMO (lento; primos pequeños)
    def __init__(self, seed: int, p: int = 10007, q: int = 10039):
        assert p % 4 == 3 and q % 4 == 3
        self.M = p * q
        x0 = seed % self.M
        if math.gcd(x0, self.M) != 1:
            x0 += 1
            while math.gcd(x0, self.M) != 1:
                x0 += 1
        self.state = x0
    def _next_bit(self) -> int:
        self.state = pow(self.state, 2, self.M)
        return self.state & 1
    def _next_uint32(self) -> int:
        v = 0
        for _ in range(32):
            v = (v << 1) | self._next_bit()
        return v
    def random(self) -> float:
        return self._next_uint32() / 2**32

class MT_from_random:
    # En CPython, random.Random usa MT19937
    import random as _r
    def __init__(self, seed: int):
        self.rng = self._r.Random(seed)
    def random(self) -> float:
        return self.rng.random()

PRNGS = {
    "MT":       MT_from_random,   # estándar Python
    "LCG":      LCG,
    "RANDU":    RANDU,
    "MiddleSq": MiddleSquare,
    "BBS":      BBS,
}

# =========================================
# Monte Carlo genérico con X ~ Unif(a,b)
# =========================================

@dataclass
class MCResult:
    I_hat: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    elapsed_s: float

def mc_integral_unif(f: Callable[[np.ndarray], np.ndarray], a: float, b: float,
                     rng_obj, n: int, alpha: float = 0.05) -> MCResult:
    # muestrar U(a,b) con cualquier rng_obj que tenga .random()
    X = np.fromiter((a + (b - a) * rng_obj.random() for _ in range(n)), dtype=float, count=n)
    Y = (b - a) * f(X)  # estimador por promedio
    t0 = time.perf_counter()
    I_hat = float(Y.mean())
    # varianza muestral (insesgada)
    s2 = float(Y.var(ddof=1)) if n > 1 else 0.0
    se = math.sqrt(s2 / n) if n > 1 else float('nan')
    z = norm.ppf(1 - alpha/2)
    ci_low, ci_high = I_hat - z*se, I_hat + z*se
    elapsed = time.perf_counter() - t0
    return MCResult(I_hat, se, ci_low, ci_high, n, elapsed)

# =========================================
# Funciones a integrar
# =========================================

def f1(x: np.ndarray) -> np.ndarray:
    # sin(pi x) en [0,1]
    return np.sin(np.pi * x)

def f2(x: np.ndarray) -> np.ndarray:
    # phi(x) = N(0,1) pdf en [0,2]
    return (1.0 / math.sqrt(2*math.pi)) * np.exp(-0.5 * x**2)

def true_values():
    I1 = 2.0 / math.pi
    I2 = norm.cdf(2.0) - 0.5
    return I1, I2

# =========================================
# Runner
# =========================================

def run_compare(n: int, seed: int, other: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    I1_true, I2_true = true_values()

    # Instanciar PRNG estándar (MT) y el otro elegido
    mt = PRNGS["MT"](seed)
    if other not in PRNGS:
        raise ValueError(f"Generador no válido: {other} (elige uno de {list(PRNGS.keys())})")
    other_rng = PRNGS[other](seed + 1)  # para que no compartan semilla exacta

    # Correr MC para ambas funciones con ambos PRNG
    rows = []
    for name, rng in [("MT", mt), (other, other_rng)]:
        # f1: ∫_0^1 sin(pi x) dx
        t0 = time.perf_counter()
        r1 = mc_integral_unif(f1, 0.0, 1.0, rng, n)
        t1 = time.perf_counter() - t0
        abs_err1 = abs(r1.I_hat - I1_true)

        # f2: ∫_0^2 phi(x) dx
        t0 = time.perf_counter()
        r2 = mc_integral_unif(f2, 0.0, 2.0, rng, n)
        t2 = time.perf_counter() - t0
        abs_err2 = abs(r2.I_hat - I2_true)

        rows.append({
            "integral": "I1=∫_0^1 sin(pi x) dx",
            "generator": name, "n": n,
            "estimate": r1.I_hat, "stderr": r1.se,
            "ci_low": r1.ci_low, "ci_high": r1.ci_high,
            "true_value": I1_true, "abs_error": abs_err1,
            "elapsed_s": t1
        })
        rows.append({
            "integral": "I2=∫_0^2 phi(x) dx",
            "generator": name, "n": n,
            "estimate": r2.I_hat, "stderr": r2.se,
            "ci_low": r2.ci_low, "ci_high": r2.ci_high,
            "true_value": I2_true, "abs_error": abs_err2,
            "elapsed_s": t2
        })

    # Guardar CSV
    csv_path = os.path.join(outdir, "mc_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Generar tabla LaTeX
    tex_path = os.path.join(outdir, "mc_results_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Tabla generada automáticamente por mc_compare.py\n")
        f.write("\\renewcommand{\\arraystretch}{1.2}\n")
        f.write("\\begin{tabular}{@{} l l r r r r r r r @{}}\n")
        f.write("\\toprule\n")
        f.write("Integral & Generador & $n$ & Estim. & EE & IC$_{95\\%}$ (inf) & IC$_{95\\%}$ (sup) & Valor real & Error abs.\\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(f"{row['integral']} & {row['generator']} & {row['n']} & "
                    f"{row['estimate']:.6f} & {row['stderr']:.6f} & "
                    f"{row['ci_low']:.6f} & {row['ci_high']:.6f} & "
                    f"{row['true_value']:.6f} & {row['abs_error']:.6f}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # Resumen consola
    print("\n=== Resultados (resumen) ===")
    for row in rows:
        print(f"{row['integral']} | {row['generator']}: "
              f"estim={row['estimate']:.6f}, EE={row['stderr']:.6f}, "
              f"IC95=({row['ci_low']:.6f},{row['ci_high']:.6f}), "
              f"real={row['true_value']:.6f}, error={row['abs_error']:.6f}, "
              f"tiempo={row['elapsed_s']:.3f}s")
    print(f"\nCSV: {csv_path}")
    print(f"LaTeX: {tex_path}")

def main():
    ap = argparse.ArgumentParser(description="Comparación MC (MT vs otro PRNG) para dos integrales.")
    ap.add_argument("--n", type=int, default=200000, help="tamaño de muestra (por PRNG)")
    ap.add_argument("--seed", type=int, default=12345, help="semilla base")
    ap.add_argument("--other", type=str, default="RANDU",
                    help=f"PRNG a comparar contra MT. Opciones: {list(PRNGS.keys())}")
    ap.add_argument("--out", type=str, default="out_mc", help="carpeta de salida")
    args = ap.parse_args()

    run_compare(n=args.n, seed=args.seed, other=args.other, outdir=args.out)

if __name__ == "__main__":
    main()