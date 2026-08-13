# Results

Google Axion Arm VM, 8 cores, 32 GiB RAM. Two images, 8-token generation,
one warm-up, and two measured repetitions.

| Deployment path | Peak RSS | Median | p95 |
| --- | ---: | ---: | ---: |
| LLaVA v1.6 Mistral-7B dual tower | 20.72 GB | 26.01 s | 29.94 s |
| SUVI uncached | 15.36 GB | 18.79 s | 18.99 s |
| SUVI cached | 15.36 GB | 18.12 s | 18.42 s |

| Threads | Uncached median | Cached median |
| ---: | ---: | ---: |
| 1 | 56.43 s | 54.98 s |
| 2 | 37.35 s | 36.50 s |
| 4 | 25.80 s | 24.76 s |
| 8 | 18.79 s | 18.12 s |

Cached SUVI preserved the same visual prefix and caption token output as the
uncached path. The LLaVA result is a same-platform resource comparison.
