# SACHS Benchmark

Real-world protein signaling network from Sachs et al. (2005). Ground truth (17 edges) is
experimentally validated via drug perturbations. This is an experiment with **real-world data**.
## Data

- **853 rows × 11 proteins** — human T-cells measured by flow cytometry (fluorescence intensity)
- Each row = one cell, one snapshot (no time series)
- Log-transform before running CI tests (values are positive, right-skewed)

Downloaded from bnlearn:
```
https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz
```
Observational condition only (no drug interventions). Loaded with `pd.read_csv(..., sep='\t')`.

## Variables

| Variable | Role |
|----------|------|
| `Raf` | Upstream MAPK signalling |
| `Mek` | RAF→MEK cascade |
| `Plcg` | Receptor-proximal signalling |
| `PIP2` | Lipid second messenger |
| `PIP3` | Lipid second messenger |
| `Erk` | Downstream of MEK |
| `Akt` | Survival signalling |
| `PKA` | Broad regulatory kinase |
| `PKC` | Receptor-proximal kinase |
| `P38` | Stress response |
| `Jnk` | Stress and apoptosis |

## Ground truth — 17 edges

From Sachs et al. (2005) Figure 2, validated by 9 drug-perturbation experiments.
Stored in `ground_truth_dag.csv`.

## Hidden confounders (for LLM elicitation)

- **EGFR/PDGFR** — upstream receptor tyrosine kinases, not measured
- **Ras GTPase** — intermediary between Raf and upstream signals
- **PTEN/PP2A** — phosphatases opposing kinase activity
- **KSR/IQGAP** — scaffold proteins organising signalling complexes

## Experimental goal

Test whether mEG-FCI with LLM-provided biochemistry knowledge recovers more of the 17-edge
network than FCI alone on 853 observational samples.

Metrics: skeleton P/R/F1, PAG-SHD, typed precision/recall vs 17-edge ground truth (paper §C.1).

## Files

| File | Description |
|------|-------------|
| `sachs_observational.csv` | 853 × 11 fluorescence measurements |
| `ground_truth_dag.csv` | 17 directed edges |

## References

Sachs K et al. (2005). Causal protein-signaling networks derived from multiparameter
single-cell data. *Science* 308(5721):523–529.
