# Code and Artifact Provenance

The public release follows the paper-facing contracts below.

| Component | Canonical source contract |
|---|---|
| Temporal GCM | v1.2 temporal GCM contract; `gcm/contract.py` is hash-pinned |
| Forecasting baselines | v1.4.1 benchmark implementation and paper hyperparameters; all v1.4.1 TDN results are excluded |
| Final TDN, H=96 | v1.5.1 unified H96 GCM-10 configuration |
| Final TDN, H=192/720 | v1.5 stability configuration `ar_b48_m050` |
| H96 selector study | v1.5.1 GCM-10, Correlation-10, MI-10, Random-10 #0, Random-10 #1 |
| H96 tuning study | v1.4.3 test-aware systematic tuning protocol |
| Published artifacts | checkpoint and metric hashes pinned in `published/CANONICAL_PROVENANCE.json` |

Published H96 GCM-10 checkpoints are shared by the main H96 benchmark and the GCM condition of the selector study. The repository stores one physical copy of each of those checkpoints.

The `published/` directory is an immutable result bundle. Training scripts write only to `results/from_scratch/` and direct TDN runners reject output paths inside `published/`.

## Canonical handoff archive

The canonical artifact handoff archive used to assemble this release had SHA256:

`9ff1b899b169f711174ad4f89aa80388a2cc288fa3974068d06a7c444c3073aa`

The Weather CSV shipped in `dataset/weather.csv` is the original frozen benchmark file with SHA256:

`f365909fa07b621e61a9a1293dc69ae803d3a865905066b481e7dcc6819932d7`
