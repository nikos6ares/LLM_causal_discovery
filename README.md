# LLM Causal Discovery

Benchmarks for **mEG-FCI**: sample-efficient constraint-based causal discovery with LLM expert guidance.

| Benchmark | Data | Ground truth | Expert knowledge |
|-----------|------|--------------|-----------------|
| `NOAH_MP/` | Synthetic (land-surface model, 10 sites, 740 rows) | Exact from model equations (17 edges) | Biophysical land-atmosphere theory |
| `SACHS/` | Real (T-cell protein signaling, 853 rows) | Experimentally validated (17 edges) | Biochemistry / signaling cascades |

See each subfolder's README for details.

## Clone

```bash
git clone https://github.com/nikos6ares/LLM_causal_discovery
cd LLM_causal_discovery
```
