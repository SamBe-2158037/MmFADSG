This is the repository pertaining to my master Thesis: "Identifying echo chambers in social networks via dense subgraph detection".

TLDR: This method uses a fair dense subgraph detection compatible with multiple groups present.

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
