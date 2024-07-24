# fair_data_representation

In this repository, we provide both the implementation of proposed algorithms and the experimental results in

"Fair Data Representation for Machine Learning at Pareto Frontier" by Shizhou Xu and Thomas Strohmer.

The experiment results are based on the following data sets:
1. Adult: UCI Adult Data Set
2. COMPAS: Correctional Offender Management Profiling for Alternative Sanctions
3. CRIME: Communities and Crime Data Set
4. LSAC: LSAC National Longitudinal Bar Passage Study data set

To derive the fair representation, we design the following ones based on the theoretical results in the paper:

For idependnet variable $X$, algorithm 1 outputs the pseudobary center.
For dependnet variable $Y$, algorithm 2 outputs the conditional pseudobarycenter.

![algorithm](https://github.com/user-attachments/assets/025cdafe-ab7b-422b-a3b0-3714472ccc24)
