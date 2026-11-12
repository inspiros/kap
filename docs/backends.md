# LAP backends

| Backend | Algorithm | Rectangular | Notes |
|---|---|---|---|
| `scipy` | modified Jonker-Volgenant | yes | default |
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

kap itself adds about 4 µs per call on top of the backend. For small
matrices, building the quotient graphs in Python costs far more than the
LAP itself.
