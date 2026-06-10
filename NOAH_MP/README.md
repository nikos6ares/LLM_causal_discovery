# NOAH-MP Benchmark

Synthetic data from the NOAH-MP land-surface model (NCAR/TN-599+STR). Ground truth DAG is
exact (derived from model equations). Expert knowledge is biophysical theory about
land-atmosphere fluxes. 

## Data

- 10 sites × 2 years hourly → subsampled to **740 rows × 12 variables**
- Forcing inputs (ERA5, causally exogenous): `Tair`, `Ur`, `SWdown`, `LWdown`, `Precip`
- Model outputs (causally endogenous): `Rn`, `LAI`, `H`, `Et`, `Eg`, `LE`, `SWC`
- Hidden (withheld from algorithm): canopy temperature `Tv`, boundary-layer resistance `rb`,
  canopy water `Wc`

**Preprocessing:**
1. Deseasonalize: subtract (hour × month) climatology to remove diurnal + annual cycles
2. Subsample every 240 h (ACF envelope method; SWC anomalies remain slow — known limitation)

## Ground truth DAG — 22 edges

Derived from NOAH-MP process equations (NCAR/TN-599+STR §3.6):

```
SWdown, LWdown, LAI → Rn       (two-stream radiation balance)
Tair, SWC           → LAI      (phenology)
Rn, Tair, Ur, LAI   → H        (aerodynamic resistance / Monin-Obukhov)
Tair, Ur, SWdown, LAI, SWC → Et (Ball-Berry stomatal conductance)
Rn, Ur, SWC         → Eg       (ground energy balance)
Et, Eg              → LE       (latent heat = λ·(Et + Eg))
Precip, Et, Eg      → SWC      (Richards equation)
```

Stored in `ground_truth_dag.csv`.

## Sites

| ID | Name | Biome |
|----|------|-------|
| 0 | Madagascar | Tropical dry forest |
| 1 | Amazon | Tropical rainforest |
| 2 | Congo | Tropical rainforest |
| 3 | Sahel | Semi-arid savanna |
| 4 | Germany | Temperate broadleaf |
| 5 | Boreal Canada | Boreal forest |
| 6 | Great Plains | Temperate grassland |
| 7 | Spain Mediterranean | Mediterranean shrubland |
| 8 | India Deccan | Tropical dry savanna |
| 9 | Australia | Semi-arid woodland |

## Experimental goal

Test whether mEG-FCI with LLM-provided biophysical knowledge recovers more of the 22-edge DAG
than FCI alone, given only ~740 samples and 3 hidden confounders.

**mEG-FCI override rule** (paper §4.1):
```
predicted_edge = expert  if  c_expert > c_FCI + δ
               = FCI      otherwise
```
Expert accuracy is swept from 0.5 → 1.0. Metrics: skeleton P/R/F1, PAG-SHD, typed
precision/recall (paper §C.1).

## Files

| File | Description |
|------|-------------|
| `causal_observed.csv` | 740 × 17 (12 vars + metadata) |
| `causal_hidden.csv` | 740 × 6 hidden variable values |
| `ground_truth_dag.csv` | 22 directed edges |
| `build_dataset.ipynb` | Full pipeline |
| `run_site.py` / `run_site.sh` | SLURM simulation pipeline |
