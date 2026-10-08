# Extraction bake-off (20 Genesis passages)

Scores are percentages. Mentions are (verse, entity) pairs after mapping each extracted name to
the answer key within its passage. Family recall is a lower bound: the answer key also lists
relations that a passage names both people for but never states. 17 answer-key links point to ids that do not exist and are ignored.

| config | n | person F1 (P/R) | place F1 (P/R) | parent F1 (P/R) | spouse F1 | sibling F1 | gender | place kind | combined | dropped | $/passage | median s | thinking |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5:low | 20 | 96.8 (99.8/93.9) | 79.2 (73.1/86.4) | 85.6 (94.4/78.4) | 82.5 | 8.7 | 98.3 | 60.0 | 87.2 | 12.2 | 0.0024 | 10.3 | 24.7 |
| claude-haiku-5-5:medium | 20 | 97.2 (99.8/94.7) | 82.5 (75.5/90.9) | 86.6 (93.6/80.5) | 81.4 | 8.0 | 98.4 | 92.9 | 88.7 | 4.7 | 0.0035 | 21.6 | 49.4 |
| claude-sonnet-5-5:low | 20 | 97.9 (99.8/96.1) | 81.3 (75.0/88.6) | 89.7 (93.2/86.4) | 79.3 | 13.2 | 98.0 | 85.7 | 89.6 | 5.0 | 0.0391 | 11.5 | 2.5 |

**Recommendation:** claude-haiku-5-5:low: the cheapest config within 3 points of the best combined score (89.6).
