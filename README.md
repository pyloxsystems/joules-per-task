# Joules Per Task

> **Correction, 2026-11-09.** The first version of this repository reported 489 J per successful task at 68% success for visual reuse plus exact incremental prefill at tau=0.05. That pairing was wrong. The 68% belongs to the tau=0.02 evaluation (34.6% skip rate); the tau=0.05 evaluation (71.7% skip rate) scored 51%. Corrected, that configuration costs about 652 J per successful task (2.124 J per step x 15,650 steps / 51 successes), 32% below the 963 J baseline, but the drop in success from 70% to 51% is significant (z about 2.75), and the figure understates the energy cost because failed episodes run longer than the baseline step count assumes. Visual reuse alone at tau=0.02 costs about 786 J per successful task at 68% success (18% below baseline, success not significantly different); reuse plus exact prefill at tau=0.02 was not measured. The claim that one-step policies are more sensitive to visual staleness is withdrawn, the 100 Wh battery figure in the paper is 374 tasks rather than 274, and synthetic inputs overstate baseline energy by 37% rather than 27%. The paper (`paper/jpt_paper.pdf`) carries the full list of changes.

Energy measurements for a vision-language-action policy on embedded hardware.

**963 J per successful task** on a Jetson Orin Nano (SmolVLA, LIBERO-Spatial, n=100),
reduced to about **786 J** (18% lower) by visual reuse at tau=0.02, with no significant change in success (68% versus 70%) and no additional training. More aggressive reuse saves more energy per step but costs success: at tau=0.05, reuse plus exact incremental prefill is about 652 J per successful task at only 51% success.

Energy is rarely reported for VLA policies. A 2026 survey of VLA efficiency metrics lists direct energy estimation as an open problem and substitutes kinematic proxies. One concurrent study (Zhou et al., arXiv:2604.24447, April 2026) measures VLA energy with a power meter on Jetson AGX Orin and Thor and reports it alongside, but not divided by, task success. This work targets the Orin Nano, the smallest Jetson module, and charges energy to successful tasks so that a configuration which saves energy by failing more often pays for its failures.

## Results

| Configuration | Success | J / successful task |
|---|---|---|
| baseline | 70% | 963 |
| visual reuse, tau=0.02 | 68% | 786 |
| visual reuse + exact incremental prefill, tau=0.05 | 51% | 652 (understated) |
| full stack (adds one-step distillation), tau=0.05 | 46% | lower, at real accuracy cost |

The tau=0.02 row is the deployable one: 18% less energy, success statistically
indistinguishable from baseline (z=0.31, n=100 per condition), no training required.
The tau=0.05 row saves 32% but loses 19 points of success (z about 2.75), and its
energy per task is understated because failed episodes run longer than the baseline
step count assumes. Reuse plus exact prefill at tau=0.02 was not measured.

Also established here:
- the failure boundary of visual reuse (success falls from 51% to 4% between 72% and 93% skip rates, against 68% at 35%)
- no evidence that one-step distilled policies are more sensitive to visual staleness than their ten-step teachers (15 versus 19 points lost at tau=0.05); an earlier version claimed they were
- six deployment failure modes, including silent corruption of prefill activations from fp16 normalization overflow under TensorRT

## Policy spectrum (added Sept 2026)

Four policies, one Jetson Orin Nano, one power rail, spanning three orders of
magnitude in parameter count:

| Policy | Parameters | Gross mJ/step | Net mJ/step |
|---|---|---|---|
| DCE | 0.2M | 10.10 | 1.55 |
| ViNT | 30M | 42.78 | 12.07 |
| NoMaD | 19M | 218.59 | 69.21 |
| SmolVLA | 450M | 4310 | - |

Gross includes the module's 6.90 W idle draw; net subtracts it. Both are given
because on this platform idle is comparable to active, and a cross-platform
comparison that does not state the convention is uninterpretable.

**The ordering is not monotonic in parameter count.** NoMaD at 19M costs 5.1x
ViNT at 30M (5.7x on net) because it runs ten sequential denoising passes per
control step. Sequential pass count, not parameter count, governs edge energy.

Applying the visual-reuse gate at a 99.5% skip rate, which is past the failure
boundary established in the paper and therefore an energy floor rather than a
deployable setting:

| Policy | Baseline | 99.5% skip | Reduction |
|---|---|---|---|
| ViNT | 42.78 mJ | 2.26 mJ | 18.9x |
| NoMaD | 218.59 mJ | 141.09 mJ | 1.55x |

The identical intervention returns 18.9x on one and 1.55x on the other. Encoder
elision pays in proportion to the encoder's share of the computation; diffusion
policies need step-count reduction instead. Raw measurements in
`paper/aerial_energy.json`, harness in `scripts/aerial/`.

Caveats kept in the open: single runs per configuration, with repeat measurement
varying 5 to 8%; no task-success number for the two navigation policies, so their
figures are per control step rather than per successful task.

## Raw measurement artifacts

- `paper/edge_logs/`: the on-device replay summaries from the Orin Nano, one file per configuration (idle and active VDD_IN, sample counts, latency per step, vision skip rate, gross and net joules per step). `measure_libero.log` is the validation run (TRT vs fp32 max deviation 2.5e-3).
- `paper/episode_steps.json`: per-episode step counts for the 100 LIBERO-Spatial episodes (15,650 steps, mean 156.5), the denominator behind joules per successful task. Baseline: 4.310 J/step x 15,650 / 70 successes = 963 J. R at tau 0.02: 3.415 x 15,650 / 68 = 786 J. R+I at tau 0.05: 2.124 x 15,650 / 51 = 652 J (the same step count is applied to every configuration, which understates energy where success is lower).
- `paper/sim_eval_summaries.txt`: the final summary line of every simulator success evaluation, with sizes and timestamps of the full logs.
- `paper/aerial_energy.json`, `paper/aerial_energy_*.txt`: the four-policy spectrum measurements.

## Method

SmolVLA (450M) decomposed into separately built TensorRT engines with an exact,
stateless per-step KV-cache contract. Engine outputs agree with fp32 reference
execution to a maximum action-space deviation of 2.5e-3. Power is sampled from the
module's INA3221 instrumentation at 10 ms intervals during replay of recorded episode
observations; task success is evaluated in the standard LIBERO simulator.

## Layout

- `paper/`: the paper (PDF and source), figures
- `scripts/`: export pipeline, measurement harness, evaluation
  - `export_engines.py`, `engines.py`: TensorRT decomposition
  - `edge_measure.py`, `edge_replay.py`: power sampling and replay
  - `eval_stack.py`, `eval_with_reuse.py`, `eval_onestep.py`: evaluation
  - `distill_lora.py`: one-step action-head distillation
  - `dump_reference.py`, `probe_onnx_export.py`: numerical validation against fp32

## Reproducing

Requires a Jetson Orin Nano (or another INA3221-instrumented board), JetPack with
TensorRT, and the LIBERO benchmark. Export the engines, validate against the fp32
reference, then run the measurement harness over recorded episodes.

## Citation

```
@misc{girard2026joules,
  title  = {Joules Per Task: Energy Measurements and an Energy-Accuracy Analysis
            for a Vision-Language-Action Policy on Embedded Hardware},
  author = {Emilio Girard},
  year   = {2026}
}
```
