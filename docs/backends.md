# LAP backends

| Backend | Algorithm | Rectangular | Notes |
|---|---|---|---|
| `kap` | shortest augmenting path (Crouse 2016) | yes | default; native, built in; same results as `scipy` |
| `scipy` | modified Jonker-Volgenant | yes | |
| `lap` | Jonker-Volgenant | yes | |
| `lapjv` | Jonker-Volgenant | no | fastest of these for n ≥ 300 |
| `lapsolver` | Jonker-Volgenant | yes | no longer builds on Python 3.12; slated for removal |
| `munkres` | Hungarian | yes | pure Python, very slow |

Measured cost of one call on a random float n × n matrix (µs, laptop, one
thread):

| n | scipy | lap | lapjv |
|---|---|---|---|
| 10 | 0.8 | 3.3 | 0.9 |
| 100 | 78 | 121 | 68 |
| 300 | 2 056 | 815 | 342 |
| 1000 | 33 542 | 24 067 | 8 844 |

`kap` (0.3.0a1, same laptop): 1.2 / 92 / 7 400 (n = 500) / 38 700 µs, within
10-35% of scipy, which uses the same algorithm; not yet tuned.

kap itself adds about 4 µs per call on top of the backend. For small
matrices, building the quotient graphs in Python costs far more than the
LAP itself.
