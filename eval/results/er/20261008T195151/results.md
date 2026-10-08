# Entity resolution scoreboard (Gen)

B³ scores each name mention by how cleanly its entity matches the real person or place the
answer key says it is. Precision falls when people are merged; recall falls when one person is
split. 17 answer-key links point to ids that do not exist and are ignored.

| Metric | gen-naive | gen-v1 |
|---|---|---|
| B³ precision | 95.9 | 97.0 |
| B³ recall | 94.8 | 99.3 |
| B³ F1 | 95.4 | 98.2 |
| Merge errors (one entity, several real people) | 7 | 5 |
| Split errors (one real person, several entities) | 3 | 0 |
| Person mention F1 | 94.1 | 95.3 |
| Place mention F1 | 74.9 | 74.6 |
| Gender accuracy | 95.2 | 95.2 |
| Place kind accuracy | 75.8 | 77.2 |
| Hard cases passed (of 6) | 2 | 6 |
| Aligned mentions | 2305 | 2305 |

## gen-naive: hard cases

| Case | Passed | Entities |
|---|---|---|
| keep lamech_1808 and lamech_1809 apart | no | per_lamech / per_lamech |
| keep enoch_1191 and enoch_1192 apart | no | per_enoch / per_enoch |
| keep nahor_2142 and nahor_2143 apart | yes | per_nahor_son_of_serug / per_nahor |
| merge all names of abraham_58 | no | per_abraham 119, per_abram 45 |
| merge all names of sarah_2473 | no | per_sarah 31, per_sarai 12 |
| merge all names of israel_682 | yes | per_jacob 170, per_israel 8 |

Largest merge errors: per_adah = adah_414 5, adah_69 3; per_lamech = lamech_1808 4, lamech_1809 4; per_anah = anah_247 3, anah_248 3, anah_249 1; per_enoch = enoch_1192 5, enoch_1191 2; pla_zoar = zoar_1271 4, bela_172 2

Largest split errors: abraham_58 = per_abraham 119, per_abram 45; sarah_2473 = per_sarah 31, per_sarai 12; nahor_2143 = per_nahor 7, per_nahor_son_of_terah 3

## gen-v1: hard cases

| Case | Passed | Entities |
|---|---|---|
| keep lamech_1808 and lamech_1809 apart | yes | per_lamech_gen_4_18 / per_lamech_gen_5_25 |
| keep enoch_1191 and enoch_1192 apart | yes | per_enoch_gen_4_17 / per_enoch_gen_5_18 |
| keep nahor_2142 and nahor_2143 apart | yes | per_nahor_son_of_serug / per_nahor |
| merge all names of abraham_58 | yes | per_abraham 164 |
| merge all names of sarah_2473 | yes | per_sarah 43 |
| merge all names of israel_682 | yes | per_jacob 178 |

Largest merge errors: pla_bethel = bethel_202 9, luz_737 3; pla_hebron = hebron_551 4, kiriath-arba_697 2, vale_of_hebron_1206 1; pla_zoar = zoar_1271 4, bela_172 2; per_zibeon = zibeon_3062 3, zibeon_3061 2; pla_bethlehem = ephrath_403 3, bethlehem_218 2

Largest split errors: 
