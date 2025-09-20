# PRNG & Monte Carlo – Reporte y Comparaciones

Este repo contiene dos programas:

1. **`prng_report.py`** — Implementa PRNGs (LCG, Middle-Square, RANDU, BBS y MT estándar de Python), genera **histogramas**, corre **pruebas de uniformidad** (chi-cuadrado) e **independencia** (corridas + autocorrelación), y compara un algoritmo elegido con `random` y `secrets`.
2. **`mc_compare.py`** — Aproxima **integrales** por Monte Carlo usando **X ~ Unif(a,b)** y compara resultados entre el generador estándar de Python (MT19937) y otro PRNG (p. ej., RANDU) **contra el valor teórico**. Exporta CSV y una **tabla LaTeX**.

---

## Requisitos

- Python 3.10+ recomendado
- Paquetes:
  - `numpy`
  - `matplotlib`
  - `scipy`

### Instalación rápida

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -U pip
pip install numpy matplotlib scipy
````

> Si compilas LaTeX, asegúrate de tener `graphicx`, `booktabs`, `tabularx`.

---

# 1) `prng_report.py` — PRNGs, histogramas y tests

### ¿Qué hace?

* Implementa **LCG**, **Middle-Square**, **RANDU**, **BBS** (demo) y **MT** (vía `random.Random`).
* Genera **N** muestras por PRNG, produce **histogramas** PNG.
* Corre pruebas:

  * **Chi-cuadrado** (uniformidad en \[0,1)).
  * **Corridas de Wald–Wolfowitz** (independencia).
  * **Autocorrelación lag-1**.
* Compara un PRNG elegido con `random` (MT) y con `secrets`.

### Cómo correr

```bash
# valores por defecto: n=100000, bins=50, out=out, compare=RANDU, seed=12345
python prng_report.py
```

Parámetros útiles:

```bash
# cambiar el algoritmo de comparación (LCG, MiddleSq, RANDU, BBS, MT)
python prng_report.py --compare LCG

# cambiar tamaño de muestra, bins y carpeta de salida
python prng_report.py --n 200000 --bins 60 --out resultados

# cambiar la semilla
python prng_report.py --seed 777
```

> **Nota BBS:** usa primos de Blum (`p % 4 == 3` y `q % 4 == 3`). En el código por defecto: `p=10007`, `q=10039`. Es **demostrativo** (primos pequeños ⇒ no cripto).

### Output

En `out/` (o la carpeta que pongas en `--out`):

* `*_hist.png` — histogramas por PRNG.
* `cmp_<ALG>_hist.png`, `cmp_MT_hist.png`, `cmp_secrets_hist.png` — histogramas de la comparación.
* `reporte.txt` — tiempos, chi-cuadrado (p-valor), corridas, autocorrelación.

### Funciones/clases (resumen)

* **`LCG`**: congruencial lineal; `random()` → `float` en \[0,1). Muy rápido, no cripto.
* **`MiddleSquare`**: método histórico; puede ciclar/colapsar. `random()` → `float`.
* **`RANDU`**: LCG histórico con mala calidad multidimensional; `random()` → `float`.
* **`BBS`**: Blum–Blum–Shub demo (lento); `random()` construye 32 bits concatenando LSBs.
* **`MT_from_random`**: envoltura de `random.Random` (MT19937); `random()` → `float`.

Utilidades:

* **`generate_samples(gen, n)`**: devuelve `np.ndarray` con `n` muestras.
* **`save_histogram(data, bins, title, path)`**: guarda histograma PNG.
* **`chi_square_uniformity(x, bins)`**: χ² vs U\[0,1).
* **`runs_test_independence(x)`**: corridas (z, p-valor).
* **`autocorr_lag1(x)`**: autocorrelación y error estándar ≈ `1/√N`.
* **`benchmark(gen_factory, n, seed)`**: genera `n` muestras y mide tiempo.
* **`summarize_tests(name, x, outdir, bins)`**: corre tests + hist y devuelve resumen de texto.

---

# 2) `mc_compare.py` — Monte Carlo para integrales y comparación

### ¿Qué hace?

* Aproxima por Monte Carlo:

  * $I_1=\int_{0}^{1}\sin(\pi x)\,dx = 2/\pi$
  * $I_2=\int_{0}^{2}\frac{1}{\sqrt{2\pi}}e^{-x^2/2}\,dx = \Phi(2)-1/2$
* Usa **X \~ Unif(a,b)** y estima **$\widehat{I}_n$**, **error estándar** y **IC 95%**.
* Compara **MT** (estándar Python) vs otro PRNG (por defecto **RANDU**) y reporta **error absoluto**.
* Exporta **CSV** y **tabla LaTeX** (`mc_results_table.tex`) lista para `\input{...}`.

### Cómo correr

* Mismo setup que antes, si ya lo hiciste solo correlo con:

```bash
# por defecto: n=200000, other=RANDU, out=out_mc, seed=12345
python mc_compare.py
```

Cambiar PRNG a comparar y tamaño de muestra:

```bash
python mc_compare.py --other LCG --n 500000
python mc_compare.py --other BBS --n 20000   # BBS es lento: usa n menor, almenos que tu PC este chetada
python mc_compare.py --out resultados_mc
```

### Output

En `out_mc/` (o `--out`):

* `mc_results.csv` — resultados por integral y generador.
* `mc_results_table.tex` — tabla LaTeX (usa `booktabs`).

### Funciones/clases (resumen)

PRNGs (interfaz `random()`):

* **`MT_from_random`, `LCG`, `RANDU`, `MiddleSquare`, `BBS`** (idénticos al otro script o equivalentes).

Monte Carlo:

* **`mc_integral_unif(f, a, b, rng_obj, n, alpha)`**
  Muestra $X_i\sim \mathrm{Unif}(a,b)$, evalúa $Y_i=(b-a)f(X_i)$, y devuelve:

  * `I_hat` (promedio), `se` (EE ≈ `s_Y/√n`), `ci_low`, `ci_high`, `n`, `elapsed_s`.

Integrandos y valores reales:

* **`f1(x)`** = `sin(pi*x)`, **`f2(x)`** = `phi(x)`
* **`true_values()`** → $(2/\pi,\ \Phi(2)-1/2)$

Runner:

* **`run_compare(n, seed, other, outdir)`**
  Ejecuta MC para MT y `other` en ambas integrales, guarda CSV y `.tex`, y imprime un resumen.

CLI:

* `--n`, `--seed`, `--other`, `--out`.

---

## Estructura

```
.
├─ prng_report.py
├─ mc_compare.py
├─ README.md
├─ out/           # generado por prng_report.py
│  ├─ *_hist.png
│  └─ reporte.txt
└─ out_mc/        # generado por mc_compare.py
   ├─ mc_results.csv
   └─ mc_results_table.tex
```