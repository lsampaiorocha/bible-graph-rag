# Benchmark against the Theographic answer key (Gen)

Scores are percentages; F1 combines precision (what we extracted that is right) and recall (what of
the key we found). Corrected columns apply eval/gold_corrections.yaml (61 verses);
17 key links point to ids that do not exist and are ignored throughout.

## Entities

| Metric | gen-naive raw | gen-naive corrected | gen-v1 raw | gen-v1 corrected |
|---|---|---|---|---|
| B³ F1 (entity resolution) | 95.4 | 95.4 | 98.2 | 98.2 |
| Merge errors | 7 | 7 | 5 | 5 |
| Split errors | 3 | 3 | 0 | 0 |
| Person mention F1 | 94.1 | 94.9 | 95.3 | 96.0 |
| Place mention F1 | 74.9 | 74.9 | 74.6 | 74.6 |
| Gender accuracy | 95.2 | 95.2 | 95.2 | 95.2 |
| Place kind accuracy | 75.8 | 75.8 | 77.2 | 77.2 |
| Hard cases passed (of 6) | 2 | 2 | 6 | 6 |

## Links (F1, with answer-key count)

| Link | gen-naive raw | gen-naive corrected | gen-v1 raw | gen-v1 corrected |
|---|---|---|---|---|
| FATHER_OF | 85.9 (287) | 86.4 (286) | 94.1 (287) | 94.6 (286) |
| MOTHER_OF | 80.0 (60) | 80.0 (60) | 84.3 (60) | 84.3 (60) |
| SPOUSE_OF | 82.9 (21) | 82.9 (21) | 90.0 (21) | 90.0 (21) |
| SIBLING_OF | 8.7 (614) | 8.7 (614) | 8.7 (614) | 8.7 (614) |
| SIBLING_OF incl. shared parents | 72.7 (614) | 72.7 (614) | 82.9 (614) | 82.9 (614) |
| BORN_IN (small sample) | 0.0 (12) | 0.0 (12) | 0.0 (12) | 0.0 (12) |
| DIED_IN (small sample) | 40.0 (8) | 40.0 (8) | 40.0 (8) | 40.0 (8) |

## Events

| Measure | gen-naive raw | gen-naive corrected | gen-v1 raw | gen-v1 corrected |
|---|---|---|---|---|
| Events found (recall) | 53.4 (62/116) | 53.4 (62/116) | 53.4 (62/116) | 53.4 (62/116) |
|   1-3 verses | 40/79 | 40/79 | 40/79 | 40/79 |
|   4-10 verses | 17/18 | 17/18 | 17/18 | 17/18 |
|   over 10 verses | 5/19 | 5/19 | 5/19 | 5/19 |
| Participant F1 (matched events) | 79.3 | 79.3 | 81.5 | 81.5 |

## Narrative links (hand-labelled)

- gen-naive: not labelled yet (eval/labels/narrative_sample_gen-naive.csv)
- gen-v1: not labelled yet (eval/labels/narrative_sample_gen-v1.csv)
