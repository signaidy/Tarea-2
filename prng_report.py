import argparse
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Optional
from scipy.stats import chisquare, norm

# =========================
# PRNGs: Implementaciones
# =========================

class LCG:
    """
    Linear Congruential Generator
    """
    def __init__(self, seed: int, m: int = 2**32, a: int = 1664525, c: int = 1013904223):
        self.m = m
        self.a = a
        self.c = c
        self.state = seed % m

    def next_uint(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def random(self) -> float:
        # Uniforme en [0,1)
        return self.next_uint() / self.m


class MiddleSquare:
    """
    Método de cuadrados medios puro
    """
    def __init__(self, seed: int, n_digits: int = 8):
        assert n_digits % 2 == 0 and n_digits >= 4
        self.n = n_digits
        self.mod = 10 ** n_digits
        self.state = seed % self.mod
        if self.state == 0:
            self.state = 10**(self.n-1) + 1  # evitar 0

    def random(self) -> float:
        y = (self.state * self.state)
        s = str(y).rjust(2 * self.n, '0')
        mid = len(s) // 2
        mid_digits = s[mid - self.n//2: mid + self.n//2]
        self.state = int(mid_digits)
        # Uniforme aproximado [0,1) sobre 10^n estados
        return self.state / self.mod


class RANDU:
    """
    RANDU clásico: X_{n+1} = (65539 * X_n) mod 2^31
    Semilla debe ser impar
    """
    def __init__(self, seed: int):
        self.m = 2**31
        self.a = 65539
        self.state = seed | 1  # forzar impar

    def next_uint(self) -> int:
        self.state = (self.a * self.state) % self.m
        return self.state

    def random(self) -> float:
        return self.next_uint() / self.m


class BBS:
    """
    Blum-Blum-Shub (demostrativo)
    """
    def __init__(self, seed: int, p: int = 10007, q: int = 10009):
        assert p % 4 == 3 and q % 4 == 3, "p y q deben ser ≡ 3 (mod 4)"
        self.M = p * q
        x0 = seed % self.M
        if math.gcd(x0, self.M) != 1:
            x0 = x0 + 1
            if math.gcd(x0, self.M) != 1:
                x0 = 3  # fallback simple
        self.state = x0

    def _next_bit(self) -> int:
        self.state = pow(self.state, 2, self.M)
        return self.state & 1  # LSB

    def _next_uint32(self) -> int:
        v = 0
        for _ in range(32):
            v = (v << 1) | self._next_bit()
        return v

    def random(self) -> float:
        # Combina 32 bits ~ U[0,1)
        return self._next_uint32() / 2**32


class MT_from_random:
    """
    En CPython, random.Random usa MT19937
    Lo envolvemos para unificar la interfaz
    """
    import random as _random
    def __init__(self, seed: int):
        self.rng = self._random.Random(seed)

    def random(self) -> float:
        return self.rng.random()


# ==================================
# Utilidades de muestreo y gráficos
# ==================================

def generate_samples(gen, n: int) -> np.ndarray:
    return np.fromiter((gen.random() for _ in range(n)), dtype=float, count=n)

def save_histogram(data: np.ndarray, bins: int, title: str, path: str):
    plt.figure()
    plt.hist(data, bins=bins, density=True)
    plt.xlabel("x")
    plt.ylabel("densidad")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# =============================
# Pruebas de hipótesis
# =============================

@dataclass
class ChiSquareResult:
    chi2: float
    pvalue: Optional[float]
    dof: int
    k: int

def chi_square_uniformity(x: np.ndarray, bins: int = 50) -> ChiSquareResult:
    """
    Test de bondad de ajuste Chi-cuadrado para U[0,1)
    """
    hist, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    expected = np.full_like(hist, fill_value=len(x) / bins, dtype=float)
    chi2 = ((hist - expected) ** 2 / expected).sum()
    dof = bins - 1
    pval = None
    # chisquare calcula automáticamente el p-valor
    cs = chisquare(hist, expected)
    chi2 = float(cs.statistic)
    pval = float(cs.pvalue)
    return ChiSquareResult(chi2=chi2, pvalue=pval, dof=dof, k=bins)


@dataclass
class RunsTestResult:
    runs: int
    z: float
    pvalue: Optional[float]
    n1: int
    n2: int
    median: float

def runs_test_independence(x: np.ndarray) -> RunsTestResult:
    """
    Test de corridas (Wald-Wolfowitz) respecto a la mediana (~0.5)
    H0: independencia (no hay patrón de corridas inusual)
    """
    med = np.median(x)
    # Secuencia binaria: 1 si >= mediana, 0 en caso contrario
    s = (x >= med).astype(int)
    # Contar corridas
    runs = 1 + np.sum(s[1:] != s[:-1])
    n1 = int(s.sum())
    n2 = int(len(s) - n1)
    if n1 == 0 or n2 == 0:
        # Degenerado: todos a un lado de la mediana
        return RunsTestResult(runs=runs, z=float('nan'), pvalue=None, n1=n1, n2=n2, median=med)

    # Estadístico aproximado normal:
    mu = 1 + (2*n1*n2) / (n1 + n2)
    sigma2 = (2*n1*n2 * (2*n1*n2 - n1 - n2)) / (((n1 + n2)**2) * (n1 + n2 - 1))
    sigma = math.sqrt(sigma2) if sigma2 > 0 else float('inf')
    z = (runs - mu) / sigma if sigma > 0 else float('nan')

    # p-valor bilateral
    pval = 2 * (1 - norm.cdf(abs(z)))

    return RunsTestResult(runs=runs, z=z, pvalue=pval, n1=n1, n2=n2, median=med)


@dataclass
class AutocorrResult:
    lag: int
    r: float
    se: float  # error estándar aprox ~ 1/sqrt(N)

def autocorr_lag1(x: np.ndarray) -> AutocorrResult:
    """
    Autocorrelación muestral en lag=1 y su error estándar ~ 1/sqrt(N)
    """
    x_mean = x.mean()
    x0 = x[:-1] - x_mean
    x1 = x[1:] - x_mean
    num = np.dot(x0, x1)
    den = np.dot(x0, x0)
    r = float(num / den) if den != 0 else float('nan')
    se = 1 / math.sqrt(len(x))
    return AutocorrResult(lag=1, r=r, se=se)


# =============================
# Bench y comparación
# =============================

def benchmark(gen_factory: Callable[[int], any], n: int, seed: int = 12345) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    gen = gen_factory(seed)
    samples = generate_samples(gen, n)
    dt = time.perf_counter() - t0
    return samples, dt

def summarize_tests(name: str, x: np.ndarray, outdir: str, bins: int = 50):
    # Histograma
    save_histogram(x, bins=bins, title=f"Histograma {name}", path=os.path.join(outdir, f"{name}_hist.png"))

    # Uniformidad
    chi_res = chi_square_uniformity(x, bins=bins)
    # Independencia
    runs_res = runs_test_independence(x)
    ac1 = autocorr_lag1(x)

    # Reporte en texto
    lines = []
    lines.append(f"== {name} ==")
    lines.append(f"N = {len(x)}")
    lines.append(f"Chi-cuadrado: chi2={chi_res.chi2:.2f}, dof={chi_res.dof}, p={chi_res.pvalue if chi_res.pvalue is not None else 'N/A (instala scipy)'}")
    lines.append(f"Runs test: R={runs_res.runs}, z={runs_res.z:.3f}, p={runs_res.pvalue if runs_res.pvalue is not None else 'N/A'}, n1={runs_res.n1}, n2={runs_res.n2}, mediana={runs_res.median:.4f}")
    lines.append(f"Autocorrelación lag1: r={ac1.r:.5f} (EE~{ac1.se:.5f})")
    lines.append("")
    return "\n".join(lines)


# =============================
# Main / CLI
# =============================

GENS = {
    "LCG":        lambda seed: LCG(seed),
    "MiddleSq":   lambda seed: MiddleSquare(seed, n_digits=8),
    "RANDU":      lambda seed: RANDU(seed),
    "BBS":        lambda seed: BBS(seed, p=10007, q=10039),
    "MT":         lambda seed: MT_from_random(seed),  # MT19937 via random.Random
    "random":     lambda seed: MT_from_random(seed),  # alias
    "secrets":    None,  # manejado aparte
}

def secrets_samples(n: int) -> np.ndarray:
    # secrets: CSPRNG del sistema; mapear enteros 0..2**32-1 a [0,1)
    import secrets
    return np.array([secrets.randbits(32) / 2**32 for _ in range(n)], dtype=float)

def main():
    parser = argparse.ArgumentParser(description="PRNG demo: histogramas y pruebas de uniformidad/independencia")
    parser.add_argument("--n", type=int, default=100_000, help="tamaño de muestra por generador")
    parser.add_argument("--bins", type=int, default=50, help="bins para histogramas/chi-cuadrado")
    parser.add_argument("--out", type=str, default="out_prng", help="directorio de salida")
    parser.add_argument("--seed", type=int, default=12345, help="semilla base")
    parser.add_argument("--compare", type=str, default="RANDU",
                        help="Algoritmo a comparar con random y secrets (opciones: LCG, MiddleSq, RANDU, BBS, MT)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report_lines = []

    # Lista base de generadores a evaluar
    eval_order = ["LCG", "MiddleSq", "RANDU", "BBS", "MT"]

    for name in eval_order:
        if name == "secrets":
            continue
        print(f"[+] Generando {name} ({args.n} muestras)...")
        if name == "MT" or name in GENS:
            samples, dt = benchmark(GENS[name], args.n, seed=args.seed)
        else:
            raise ValueError(f"Generador desconocido: {name}")
        report_lines.append(f"Tiempo {name}: {dt:.3f} s ({args.n/dt:,.0f} nums/s)")
        report_lines.append(summarize_tests(name, samples, args.out, bins=args.bins))

    # Comparación pedida: algoritmo vs random (MT) y secrets
    cmp_name = args.compare
    if cmp_name not in GENS:
        raise ValueError(f"--compare debe ser uno de: {', '.join(['LCG','MiddleSq','RANDU','BBS','MT'])}")

    print(f"[+] Comparación: {cmp_name} vs random (MT) vs secrets")
    # Algoritmo elegido
    x_alg, dt_alg = benchmark(GENS[cmp_name], args.n, seed=args.seed)
    # random (MT)
    x_mt, dt_mt = benchmark(GENS["MT"], args.n, seed=args.seed + 1)
    # secrets (CSPRNG)
    t0 = time.perf_counter()
    x_sec = secrets_samples(args.n)
    dt_sec = time.perf_counter() - t0

    # Guardar histogramas
    save_histogram(x_alg, args.bins, f"Histograma {cmp_name}", os.path.join(args.out, f"cmp_{cmp_name}_hist.png"))
    save_histogram(x_mt,  args.bins, f"Histograma random(MT)", os.path.join(args.out, f"cmp_MT_hist.png"))
    save_histogram(x_sec, args.bins, f"Histograma secrets",   os.path.join(args.out, f"cmp_secrets_hist.png"))

    # Resumen
    report_lines.append("== Comparación de velocidad ==")
    report_lines.append(f"{cmp_name}: {dt_alg:.3f} s ({args.n/dt_alg:,.0f} nums/s)")
    report_lines.append(f"random (MT): {dt_mt:.3f} s ({args.n/dt_mt:,.0f} nums/s)")
    report_lines.append(f"secrets: {dt_sec:.3f} s ({args.n/dt_sec:,.0f} nums/s)")
    report_lines.append("")

    report_lines.append(summarize_tests(f"CMP_{cmp_name}", x_alg, args.out, bins=args.bins))
    report_lines.append(summarize_tests("CMP_MT", x_mt, args.out, bins=args.bins))
    report_lines.append(summarize_tests("CMP_secrets", x_sec, args.out, bins=args.bins))

    # Guardar reporte de texto
    report_path = os.path.join(args.out, "reporte.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n[✓] Listo.")
    print(f" - Carpeta de salida: {args.out}")
    print(f" - Reporte: {report_path}")
    print(f" - Imágenes: *_hist.png")


if __name__ == "__main__":
    main()