# Inkling 2 Mini bullet-margin results

The released rank-eight adapter was trained with the exact reviewed Particle
Sliders Core 0.14.4 wheel and ParticleGAN 0.11.0 using explicit stock EMPF,
scoped QR and R2 initialization, and no weight EMA. The seven rate arms
trained for 64 updates each. The bank has 128 particles with four coordinates.

The rank statistic is the equal-case mean gain in distinct Markdown bullet
lines, capped at three, from matched ON and OFF native outputs. The six-case
panel used four full-strength and two half-strength comparisons. Catastrophic
output integrity was evaluated separately. All 63 saved ON outputs passed
the integrity checks.

| Candidate | Generator | Critic | Bank | Mean gain |
| --- | ---: | ---: | ---: | ---: |
| Stock, released RAW064 | 0.012 | 0.018 | 0.030 | +1 |
| Generator half | 0.006 | 0.018 | 0.030 | +1/3 |
| Generator double | 0.024 | 0.018 | 0.030 | +17/18 |
| Critic half | 0.012 | 0.009 | 0.030 | +1 |
| Critic double | 0.012 | 0.036 | 0.030 | +1 |
| Bank half | 0.012 | 0.018 | 0.015 | +1 |
| Bank double | 0.012 | 0.018 | 0.060 | +1 |

Generator 0.012 scored highest with critic and bank held fixed. All five
maximal tuples share that generator rate. Stock is the independently
replayed example from the tie.

[Rates and matched outputs](evidence/lr-results.json) ·
[Exact-rational ranking and checkpoint audit](evidence/checkpoint-audit.json)

## Independent examples

A fresh process using the installed wheel generated all three new examples
and reproduced the earlier baking-spices ON and OFF token sequences exactly.

| Prompt topic | OFF | ON |
| --- | --- | --- |
| Fabric | Cotton, silk, wool. | - Cotton<br>- Silk<br>- Wool |
| Autumn leaves | Red, orange, yellow. | - Red<br>- Orange<br>- Yellow |
| Freshwater fish | Trout, carp, and perch. | - Trout<br>- Bass<br>- Carp |

[Complete examples with token IDs](evidence/working-examples.json)

## Why the first slider missed

The first cohort matched the hidden state before generation. In the native
token readout, the opening dash was behind the best competing token by
4.97 logits OFF and 5.02 with stock RAW008. The explicit bullet teacher
put the dash ahead by 17.47 logits. Giving the frozen base just one initial
dash led it to generate three complete bullets in all four cases.

The follow-up learned the dash margin on shared answer prefixes at all three
bullet positions, with neutral targets inside lines. The fit changed the
target, adapter attachment point, and update budget together.

[Held-out opening-token diagnostic](evidence/initial-heldout-logits.json) ·
[Training-prompt diagnostic](evidence/initial-training-logits.json)
