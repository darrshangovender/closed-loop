# Platforms (SYNTHETIC)

Both full runs (2 and 3 November 2026) were on one Windows 10 laptop: Python 3.12.13, numpy 2.5.4,
scipy 1.18.1, as the ledger records. The second run reused the first run's cached search cells, so
the search stage (C-SEARCH) had been computed once.

## C-SEARCH recomputed on Linux (16 November 2026)

GitHub Actions `ubuntu-latest` (x86_64, glibc 2.43), Python 3.12.16, numpy 2.5.4, scipy 1.18.1,
4 workers, an empty cache, the code at this repository's `main` (nothing under `closed_loop/` has
changed since v0.1.0). The seeds are the published ones, `search_seed(pre-registration hash, i)` for
i = 0 to 9, derived without the held-out guard, so nothing was written to the ledger. The same script
pointed at a copy of the Windows cache reproduces all 25 records of run 2, which checks the script.

21 of the 25 C-SEARCH records are identical in value, interval and sample size. The 4 that differ
are all CMA-ES inside R_in:

| Record | Windows (published) | Linux |
|---|---|---|
| `search.R_in.cma.find_rate` and `claim/C-SEARCH/find_rate/R_in/cma` | 6 of 10 | 7 of 10 |
| `search.R_in.cma.contract_failure_runs` | 6 of 10 | 7 of 10 |
| `search.R_in.cma.evals_to_first` (mean, budget 200) | 97.0 | 79.9 |

Identical on both: every random-search record, every R_out record (CMA-ES and random search each
found a violation in 10 of 10 runs; the pairs CMA-ES 5, random search 2, 3 dropped), the defect
counts (`search.R_in.*.defect_runs` 0 of 10, `claim/C-SEARCH/r_in_violations` and
`claim/C-SHIELD/search_r_in` 0 of 20), and so the C-SEARCH verdict, which is decided on the R_out
pairs alone: inconclusive on both platforms.

Why: CMA-ES samples along the eigenvectors of its covariance matrix (`numpy.linalg.eigh`). Different
LAPACK builds and CPUs return different but equally valid eigenvectors, so from the same seed the
runs visit different points. Random search draws its points directly from the seed and does not
depend on the platform. The list of R_in contract failures in
[WHAT-GETS-THROUGH](WHAT-GETS-THROUGH.md) comes from the Windows run; a CMA-ES run on another machine
can find a different set.

## The unit test this showed up in

`tests/search/test_falsify.py` plants a violation in a small corner of the box and checks that
CMA-ES reaches it within 200 evaluations and random search does not. Over seeds 3000 to 3099:

| Machine | Python | numpy / scipy | CMA-ES reached it | of which in seeds 3000 to 3009 | Random search |
|---|---|---|---|---|---|
| Windows laptop | 3.11.15 | 2.4.6 / 1.17.1 | 38 of 100 | 5 of 10 | 0 of 100 |
| Windows laptop | 3.12.13 | 2.5.4 / 1.18.1 | 29 of 100 | 3 of 10 | 0 of 100 |
| Linux runner | 3.11.18 | 2.4.6 / 1.17.1 | 25 of 100 | 3 of 10 | 0 of 100 |
| Linux runner | 3.12.16 | 2.4.6 / 1.17.1 | 30 of 100 | 0 of 10 | 0 of 100 |
| Linux runner | 3.12.16 | 2.5.4 / 1.18.1 | 30 of 100 | 2 of 10 | 0 of 100 |

The public CI's first Linux 3.11 job reached it on 0 of the first 10 seeds; a later 3.11 job on a
different host, on 3 of 10. The test used to ask for 2 of the first 10; it now asks for 15 of the
100 and none for random search.
