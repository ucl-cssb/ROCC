# Provenance

This repository is a fork of ROCC, used as the baseline for my coursework in
Reinforcement Learning for Bioengineers (BIOE70077), Imperial College London, 2026-27.

## Source paper

Treloar NJ, Fedorec AJH, Ingalls B, Barnes CP (2020).
Deep reinforcement learning for the control of microbial co-cultures in bioreactors.
*PLoS Computational Biology* 16(4): e1007783.
https://doi.org/10.1371/journal.pcbi.1007783

Data for the paper's main figures: Zenodo, https://doi.org/10.5281/zenodo.3728079

## Source code

| Item | Value |
|---|---|
| Upstream repository | https://github.com/ucl-cssb/ROCC |
| Branch | `master` |
| Starting commit | `e7491672bcafc0fac08fe750829e4fac2805d35a` (short: `e749167`) |
| Commit date | 2020-11-17 |
| Commit author | Neythen Treloar |
| Forked on | 2026-11-21 |
| Git tag for the untouched baseline | `baseline-e749167` |
| Licence | MIT, copyright (c) 2020 Computational Systems and Synthetic Biology, UCL. See `LICENSE`. |

Everything at tag `baseline-e749167` is the original authors' work, unchanged.
Every change I make comes after that tag and can be listed with:

```bash
git log baseline-e749167..HEAD
git diff baseline-e749167 HEAD
```

## Software versions reported in the paper

Python 3.6.7, SciPy 1.3.1, TensorFlow 1.13.1, NumPy 1.16.14 (as printed; no such
release exists, the 1.16 series ended at 1.16.6), MATLAB R2019a for PI controller tuning.
Hardware and compute time are not reported.

## Known differences between the paper and this code

Found while reading the paper against commit `e749167`. Recorded here so the
reproduction can account for them.

1. **Yield coefficients.** Paper Table 1 gives γ0 = 4.8, γ1 = 5.2, γ2 = 4.4 (×10^11 cells/g).
   `ROCC/chemostat_env/parameter_files/double_aux.yaml` uses 4.8 for both auxotrophic
   nutrients and 5.2 for the carbon source (scaled units).
2. **Terminal penalty.** `reward_func.py` returns -1 and ends the episode if any
   population falls below 1000 (10^9 cells/L). The paper does not mention this.
3. **Fig 3 metric.** The paper reports a sum of squared errors; `get_sum_squared_error`
   in `examples/Figure_3/PI_comp_example.py` sums absolute errors.
4. **Exploration schedule.** The paper gives ε = 1 − log10(aE) with a = 0.3 and also calls
   the decay "exponential". The code uses ε = 1 − log10((E+1)/3), clipped to [0, 1].
5. **Interaction term.** The code's ODE includes a species interaction matrix A
   (`dN = N(μ + AN − q)`), set to zero. The paper's model has no interaction term.
6. **TensorFlow version.** The README says TensorFlow 2.1.2 and lower should work, but
   `reset_weights()` in `fitted_Q_agents.py` uses TensorFlow 1 APIs
   (`tf.keras.backend.get_session`, `tf.global_variables_initializer`).
7. **Example defaults.** `examples/Figure_2/double_aux_example.py` sets `n_episodes = 1`;
   the paper trains for 30 episodes.

## Attribution

If you reuse this work, cite the source paper above and the upstream repository.
My own additions are described in `README.md` as the project develops.
