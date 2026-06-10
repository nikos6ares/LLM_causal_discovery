# NOAH-MP Benchmark

Synthetic data from the NOAH-MP land-surface model (NCAR/TN-599+STR). Each row is a
simultaneous snapshot of all 12 variables at a single hourly timestep. The ground truth
DAG captures which variables structurally determine which others **within the same timestep**,
as given by the NOAH-MP process equations.

## Data

- 10 sites × 2 years of hourly NOAH-MP output → subsampled to **740 rows × 12 variables**
- Each row = one hourly snapshot; all variables observed at the same time `t`

**Variable types:**

| Type | Variables | Description |
|------|-----------|-------------|
| Forcing (exogenous) | `Tair`, `Ur`, `SWdown`, `LWdown`, `Precip` | Prescribed from ERA5; not computed by model |
| State (pre-existing) | `SWC`, `LAI` | Soil water content and leaf area index; carried over from previous timestep |
| Fluxes (computed) | `Rn`, `H`, `Et`, `Eg`, `LE` | Computed within this timestep from forcing + state |

**Hidden (withheld from algorithm):** canopy temperature `Tv`, boundary-layer resistance `rb`,
canopy water `Wc` — these mediate causal paths but are not in the data.

## Preprocessing

Raw hourly time series are not i.i.d.: consecutive rows share the same seasonal cycle and
are strongly autocorrelated. Two steps fix this:

1. **Deseasonalize** — subtract the (hour-of-day × month) climatological mean from each
   variable, removing diurnal and annual cycles. Residuals reflect synoptic anomalies.
2. **Subsample every 240 h** — ACF envelope method (tracking local maxima of |ACF| to
   avoid false zero-crossings from oscillatory signals) shows decorrelation lags up to
   ~240 h for most variables. Subsampling at this interval gives near-independent rows.
   SWC anomalies remain autocorrelated beyond 240 h — known limitation.

Result: **740 rows** (74 per site × 10 sites), treated as i.i.d.

## Ground truth DAG — 17 edges

Only within-timestep structural dependencies. Cross-timestep state updates
(`Precip→SWC`, `Tair/SWC→LAI`, `Et/Eg→SWC`) are excluded — they describe how the
model evolves over time, not what determines what within a single snapshot.

```
SWdown, LWdown, LAI      → Rn     (radiation balance)
Rn, Tair, Ur, LAI        → H      (sensible heat, via hidden Tv, rb)
Tair, Ur, SWdown, LAI, SWC → Et   (transpiration, via hidden rb, β, rs)
Rn, Ur, SWC              → Eg     (ground evaporation, via hidden rsurf)
Et, Eg                   → LE     (latent heat = λ·(Et + Eg))
```

`SWC` and `LAI` are state variables — they have no within-snapshot causal parents
but influence the flux computation.

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

Test whether mEG-FCI with LLM-provided biophysical knowledge recovers more of the 17-edge
DAG than FCI alone, given ~740 samples and 3 hidden confounders.

**mEG-FCI override rule** (paper §4.1):
```
predicted_edge = expert  if  c_expert > c_FCI + δ
               = FCI      otherwise
```
Expert accuracy swept 0.5 → 1.0. Metrics: skeleton P/R/F1, PAG-SHD, typed precision/recall.

## Files

| File | Description |
|------|-------------|
| `causal_observed.csv` | 740 × 17 (12 vars + metadata) |
| `causal_hidden.csv` | 740 × 6 hidden variable values |
| `ground_truth_dag.csv` | 17 directed edges |
| `build_dataset.ipynb` | Full pipeline |
| `run_site.py` / `run_site.sh` | SLURM simulation pipeline |
| `sites.tar.gz` | Raw per-site HRLDAS output (compressed) |
| `technotes_599.pdf` | NCAR/TN-599+STR: NOAH-MP v5.0 technical description |

## References

Yang Z-L et al. (2023). NCAR/TN-599+STR.
Spirtes et al. (2000). *Causation, Prediction, and Search*. MIT Press.
