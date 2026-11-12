# LAP backends

| Backend | Algorithm | Rectangular | Notes |
|---|---|---|---|
| `kap` | Jonker-Volgenant initialisation (square) + shortest augmenting path (Crouse 2016) | yes | default; native, built in; optimal like `scipy` (ties may differ on square input) |
| `scipy` | modified Jonker-Volgenant | yes | |
| `lap` | Jonker-Volgenant | yes | |
| `lapjv` | Jonker-Volgenant | no | fastest of these for n ≥ 300 |
| `lapsolver` | Jonker-Volgenant | yes | no longer builds on Python 3.12; slated for removal |
| `munkres` | Hungarian | yes | pure Python, very slow |

Measured cost of one call, in µs (server4, one pinned core, kap 0.3.0a1,
`develop/bench_lap.py`; uniform = random floats in [0, 1), int100 = integers in
[0, 100) with many ties, geometric = `i*j + noise`, a hard structured case):

| Input | n | kap | scipy | lap | lapjv |
|---|---|---|---|---|---|
| uniform | 10 | 2.0 | 1.9 | 2.4 | 2.3 |
| uniform | 100 | 112 | 340 | 106 | 67 |
| uniform | 300 | 1 729 | 3 789 | 1 878 | 1 272 |
| uniform | 1000 | 20 164 | 72 584 | 42 789 | 29 391 |
| uniform | 2000 | 96 435 | 351 510 | 134 195 | 105 232 |
| int100 | 1000 | 22 954 | 33 742 | 5 304 | 5 810 |
| geometric | 1000 | 737 934 | 773 744 | 2 326 726 | 7 385 561 |
| uniform, n × 2n | 1000 | 20 172 | 19 502 | 28 547 | — |

kap is the fastest on large uniform and structured square matrices, and stays
within 0.8-1.0x of scipy on rectangular ones. Square matrices with many ties
are still 2.5-4x slower than lap/lapjv (work in progress).

kap itself adds about 4 µs per call on top of the backend. For small
matrices, building the quotient graphs in Python costs far more than the
LAP itself.
