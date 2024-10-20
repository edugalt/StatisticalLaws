# Statistical Laws in Complex Systems

This repository contains the data and code used in: 
- [E. G. Altmann](https://www.maths.usyd.edu.au/u/ega/), "Statistical Laws in Complex Systems: Combining Mechanistic Models and Data Analysis", <br>

[monograph (expected Dec. 2024)](https://link.springer.com/book/9783031731631) part of the Springer series [”Understanding Complex Systems”](https://www.springer.com/series/5394), pre-print available at [arXiv:2407.19874](https://arxiv.org/abs/2407.19874) (Jul. 2024).

## Folders:

- [data/](https://github.com/edugalt/StatisticalLaws/blob/main/data/) <br>
datasets us in the analysis of the different Statistical Laws:
   - [Allometric laws](https://github.com/edugalt/StatisticalLaws/tree/main/data/allometric)
   - [Cities](https://github.com/edugalt/StatisticalLaws/tree/main/data/cities)
   - [Income (Pareto's law)](https://github.com/edugalt/StatisticalLaws/tree/main/data/income)
   - [Linguistic laws](https://github.com/edugalt/StatisticalLaws/tree/main/data/language) Large datasets, needs to be downloaded from https://doi.org/10.5281/zenodo.13119897

- [notebooks/](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/) <br>
jupyter notebooks used to generate the figures and tables of the [paper](https://arxiv.org/abs/2407.19874):

  - [allometric.ipynb](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/allometric.ipynb) contains the analysis of Kleiber’s law and allometric scaling laws – Sec. 2.2.3 – including Figs. 2.7 and 2.8.
  - [bibliometric-data.ipynb](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/bibliometric-data.ipynb) contains the analysis of the bibliometric data shown in Fig. 4.2.
  - [burstinessWords.ipynb](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/burstinessWords.ipynb) contains the analysis of the inter-event time between words – Sec. 2.3.1 – including Figs. 2.11 and 3.1.
  - [cities.ipynb contains the analysis of all urban data, including the ALZ law – Sec. 2.1.2 –, urban scaling laws – Sec. 2.2.1 –, Figs. 1.1, 2.2, 2.5, 3.2, 3.3, 3.5, and 3.7, and Tab. 3.4.3.
  - [constrained-powerlaw.ipynb](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/constrained-powerlaw.ipynb) contains the code to generate constrained surrogates – Sec. 3.4.2 – including Fig. 3.12.
  - [heaps.ipynb](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/heaps.ipynb) contains the analysis of Herdan-Heaps’ law – Sec. 2.2.2 – including Fig. 2.6.
  - [pareto.ipynb](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/) contains the analysis of Pareto’s law of inequality – Sec. 2.1.1 – including Fig. 2.1
  - [synthetic-powerlaw.ipynb](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/synthetic-powerlaw.ipynb) contains the generation and analysis of synthetic power-law datasets with correlation – Sec. 3.3.4 – including Figs. 3.8 and 3.9.
  - [zipf.ipynb](https://github.com/edugalt/StatisticalLaws/blob/main/notebooks/zipf.ipynb) Contains the analysis of Zipf’s law of word frequencies – Sec. 2.1.3 – including Figs. 2.3-3.6 and Tab. 3.3-3.4

- [src/](https://github.com/edugalt/StatisticalLaws/blob/main/src/) <br>
source code used in the data analysis.

 
## Credit:

This repository builds on the ideas, code, and data from:

- Urban Scaling laws:
  - Paper: J. C. Leitao, J.M. Miotto, M. Gerlach, and E. G. Altmann, "Is this scaling nonlinear?", [Royal Society Open Science 3, 150649 (2016)](http://rsos.royalsocietypublishing.org/content/3/7/150649) 
  - Paper: E. G. Altmann, "Spatial interactions in urban scaling laws", [PLOS ONE 15, e0243390 (2020)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243390) 
  - Code: https://github.com/edugalt/scaling

- Fitting frequency distributions and rank-frequency distributions:
  -  Paper: M. Gerlach and E. G. Altmann, "Stochastic model for the vocabulary growth in natural languages", [Phys. Rev. X 3, 021006 (2013)](http://link.aps.org/doi/10.1103/PhysRevX.3.021006)
  -  Paper: H. H. Chen, T. J. Alexander, D. F.M. Oliveira, E. G. Altmann, "Scaling laws and dynamics of hashtags on Twitter", [Chaos 30, 063112 (2020)](https://doi.org/10.1063/5.0004983) or [arXiv](https://arxiv.org/abs/2004.12707)
  -  Code: https://github.com/edugalt/TwitterHashtags

- Effect of correlation
  - Paper: M. Gerlach and E. G. Altmann, "Testing statistical laws in complex systems", [Phys. Rev. Lett. 122, 168301 (2019)](https://doi.org/10.1103/PhysRevLett.122.168301) or [arXiv](https://arxiv.org/abs/1904.11624)
  - Code: https://github.com/martingerlach/testing-statistical-laws-in-complex-systems

- Constrained surrogates
  - Paper: J. M. Moore, G. Yan, E. G Altmann, "Nonparametric Power-Law Surrogates", [Phys. Rev. X 12, 021056 (2022)](https://doi.org/10.1103/PhysRevX.12.021056)
  - Code: https://github.com/JackMurdochMoore/power-law/



