This is the repository pertaining to my master Thesis: "Identifying echo chambers in social networks via dense subgraph detection".

TLDR: This method uses a fair dense subgraph detection compatible with multiple groups present to identify echo chambers present in polarized social graphs.

## Abstract:
Online social networks have been widely implicated as a source for the rise of polar-
ization. Echo chambers, which are tightly connected communities where like minded
information can flow freely but opposing views cannot penetrate, are identified as
the central mechanism that facilitate polarization. A natural way to detect for such
structures is by searching for dense regions of a network. However, classic dense
subgraph mechanisms will only recover one tight knit corner of the polarized graph,
yielding a single group rather than the entire echo chamber structure. Existing for-
mulations address only the bipartisan case, or lack viability on a larger scale. Our
formulation, which explores a multi-group setting that is typically present within
European politics remains largely unexplored.
This thesis introduces the Max-min Fairness Aware Densest Subgraph Problem
which aims to find a subgraph that is both dense, and has a minimum represented
proportion across an arbitrary number of subgroups within a partition. The min-
imum operator however renders the objective intractable, motivating the use for a
soft-min operator as a replacement. The Log Sum Exp operator has been used which
is unfortunately neither super nor submodular, ruling out using off the shelf solvers
that require this structure. Our solution to this was to characterize these super and
submodular directions and construct a surrogate that decomposes this into a super-
modular and submodular function, where the submodular part can be modularized.
This inner approximation is supermodular and can be solved using iterative peeling.
Composing this with the existing Submodular supermodular framework yields an
algorithm that has a non-decreasing guarantee across steps.
The methods were evaluated on real-world political networks, with the Belgian
Chamber of Representatives as the central talking point. The experiments illustrate
how approximation methods and parameter selection shape the regularization path,
and show that a target level of minimum represented proportion can be recovered.

## Requirements
 
Python 3.9+ with:
 
```bash
pip install numpy networkx matplotlib tqdm pandas seaborn
```
 
`pandas` and `seaborn` are only needed for the plotting notebook.
 
## Data layout
 
Datasets are read from a `datasets/` directory relative to the working directory:
 
```
datasets/
├── pol_books/polbooks.gml
└── cosponsor/
    ├── Bel/net_be_ch2014.gml
    └── It/net_it_ca2001.gml
```
 
Supported `--dataset-name` values: `polbooks`, `cosponsorBel`, `cosponsorIt`.\\
The CosponsorBel and cosponsorIt datasets are cosponsor networks from politicians in the lower chamber repectively of Italy and Belgium.
This dataset can be found [Here](https://github.com/briatte/parlnet)\\
The Political books dataset represents books that are frequently bought together, which can be found [Here](https://networkrepository.com/polbooks.php)\\
Both datasets are properly cited in the main text.
 
## Running the regularization path
 
The main experiment sweeps `λ` over a grid and logs the path to
`logs/<dataset>/<dataset>_log.npy`:
 
```bash
python reg_path_partial_linearization.py --dataset-name cosponsorBel
```
 
Key arguments:
 
| Argument | Default | Meaning |
|----------|---------|---------|
| `--dataset-name` | `cosponsorBel` | Dataset to load. |
| `--mu` | `3` | LSE temperature (larger ⇒ closer to a hard `min` over groups). |
| `--alpha` | `None` | DC correction for `Q_cross`; `None` ⇒ `default_alpha(mu, L)`. |
| `--lam-max` | `200.0` | Upper end of the `λ` grid. |
| `--num-lam` | `160` | Number of `λ` points. |
| `--num-passes` | `5` | SuperGreedy++ passes per inner run. |
| `--num-outer` | `10` | SSP outer (re-linearization) iterations per `λ`. |
| `--linearization-method` | `2` | Linearization scheme (`1`–`4`, see below). |
| `--initial-graph` | `0` | Warm-start: `0` cold (from `G`), `1` from the unfair densest subgraph, `2` warm-start along the path. |
 
Example with a finer grid and warm-start:
 
```bash
python reg_path_partial_linearization.py \
    --dataset-name cosponsorBel \
    --mu 2.0 --lam-max 400 --num-lam 300 \
    --num-outer 20 --initial-graph 2
```
 
**Linearization methods:** `1` finite-difference marginals,
`2` block ordering ascending , `3` symmetric subgradient
, `4` block ordering descending.
 
## Bisection to a target diversity
 
To binary-search `λ` until the minimum represented proportion reaches a target
level (within `--epsilon`):
 
```bash
python run_bisection_linearization.py --dataset-name cosponsorBel --target_alpha 0.1
```
 
`run_bisection_linearization_sweep.py` repeats this across a range of targets
(`0 … 1/L`) and plots reached vs. target diversity.
 
## Plotting
 
Open `plot_reg_path.ipynb`, set `dataset_name` (and `L`) in the first cells,
and run them to render the regularization-path figures, the μ/α-sensitivity and
warm-start sweeps, and the network visualizations.
 
## Output format
 
The sweep saves a pickled dict; load it with:
 
```python
import numpy as np
d = np.load("logs/cosponsorBel/cosponsorBel_log.npy", allow_pickle=True).item()
```
 
Useful keys include `lam`, `protected_portion_in_sub` (the minimum represented
proportion, the main fairness curve), `density`, `cost_value` (full fair
objective), `num_of_nodes`, `group_sizes`, and the parameters `mu`/`alpha`.

