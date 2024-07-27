# fair_data_representation

In this repository, we provide both the implementation of proposed algorithms and the experimental results in

"Fair Data Representation for Machine Learning at Pareto Frontier" [[1]](#1) by Shizhou Xu and Thomas Strohmer.

## Goal:

In a nutshell, the project aims to provide a mathematically rigorous framework for how to provably remove the sensitive information contained in some feature variables directly from data at the lowest possible utility cost.

## Datasets we tested:

The experiment results are based on the following data sets:
1. Adult: UCI Adult Data Set
2. COMPAS: Correctional Offender Management Profiling for Alternative Sanctions
3. CRIME: Communities and Crime Data Set
4. LSAC: LSAC National Longitudinal Bar Passage Study data set

## Algorithm design for tabular data:

To derive the fair synthetic tabular data, we design the following ones based on the theoretical results in the paper:

For idependnet variable $X$, algorithm 1 outputs the pseudo-barycenter.
For dependnet variable $Y$, algorithm 2 outputs the conditional pseudo-barycenter.

![algorithm](https://github.com/user-attachments/assets/025cdafe-ab7b-422b-a3b0-3714472ccc24)

## Implementation in Python:

The following is a Python implementation of the Pseudo-barycenter function, which takes in

1. data = data_set (excluding sensitive feature)
2. sensitive = sensitive feature or variable
3. threshold = the stoping criteria for estimating the barycenter covariance matrix

and output

1. X_bar = the desensitized version of the original training set, which has theoretical guarantee for the first two moments to satisfy strict statistical parity
2. OT_mp = a list (ordered by the sensitive feature or variable) of linear transport maps
3. X_mean = a list (ordered by the sensitive feature or variable) of the sensitive group averages/means
4. X_ave = the original data set average/mean

Thereafter, we can use 2, 3, and 4 to generate the optimal affine map and McCann interpolation for testing set.

```python
import numpy as np
from scipy.linalg import sqrtm

def PseudoBarycenter(data, sensitive, threshhold):
    X_mean = []
    X_cov = []
    OT_Map = []
    X_dim = data.shape[1]
    Z_range = len(set(sensitive))
    X_ave = np.average(data, axis = 0)

    for i in range(Z_range):
        X_mean.append(np.average(data[sensitive == list(set(sensitive))[i]], axis = 0))
        X_cov.append(np.cov(data[sensitive == list(set(sensitive))[i]].T))

    X_barcov = np.random.rand(X_dim,X_dim)
    eps = 1000
    while eps > threshhold:
        X_new = np.zeros((X_dim, X_dim))
        for i in range(Z_range):
            X_new = X_new + (1/Z_range) * sqrtm(sqrtm(X_barcov) @ X_cov[i] @ sqrtm(X_barcov))
        eps = np.linalg.norm(X_new - X_barcov)
        X_barcov = X_new

    X_bar = np.zeros(data.shape)
    for i in range(Z_range):
        transport = np.linalg.inv(sqrtm(X_cov[i])) @ sqrtm( sqrtm(X_cov[i]) @ X_barcov @ sqrtm(X_cov[i]) ) @ np.linalg.inv(sqrtm(X_cov[i]))
        OT_Map.append(transport)
        X_bar[sensitive == list(set(sensitive))[i]] = (data[sensitive == list(set(sensitive))[i]] - X_mean[i]) @ transport.T + X_ave

    return X_bar, OT_Map, X_mean, X_ave
```

If you have questions, segguestions, or want to discuss more details, please contact me via shzxu(at)ucdavis(dot)edu

## References
```latex
@article{JMLR:v24:22-0005,
  author  = {Shizhou Xu and Thomas Strohmer},
  title   = {Fair Data Representation for Machine Learning at the Pareto Frontier},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {331},
  pages   = {1--63},
  url     = {http://jmlr.org/papers/v24/22-0005.html}
}
```
