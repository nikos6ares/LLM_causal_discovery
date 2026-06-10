# LLM Causal Discovery

Benchmarks for **mEG-FCI**: sample-efficient constraint-based causal discovery with LLM expert guidance.

| Benchmark | Data | Ground truth | Expert knowledge |
|-----------|------|--------------|-----------------|
| `NOAH_MP/` | Synthetic (land-surface model, 10 sites, 740 rows) | Exact from model equations (22 edges) | Biophysical land-atmosphere theory |
| `SACHS/` | Real (T-cell protein signaling, 853 rows) | Experimentally validated (17 edges) | Biochemistry / signaling cascades |

See each subfolder's README for details.

## Clone (with large files)

`NOAH_MP/sites.tar.gz` and `NOAH_MP/technotes_599.pdf` are stored via Git LFS.

```bash
# 1. install git-lfs (once per machine)
conda install -c conda-forge git-lfs -y

# 2. activate lfs hooks (once per machine)
git lfs install

# 3. clone — LFS files download automatically
git clone https://github.com/nikos6ares/LLM_causal_discovery
cd LLM_causal_discovery
```
