# Joules Per Task

Energy measurements for a vision-language-action policy on embedded hardware.

**963 J per successful task** on a Jetson Orin Nano (SmolVLA, LIBERO-Spatial, n=100),
reduced to **489 J** with no measurable accuracy cost and no additional training.

These appear to be the first hardware energy measurements reported for a VLA policy
on a standard manipulation benchmark. A 2026 survey of VLA efficiency metrics lists
direct energy estimation as an open problem and substitutes kinematic proxies; a
review of sixteen recent efficiency papers found none reporting energy in any form.

## Results

| Configuration | Success | J / successful task |
|---|---|---|
| baseline | 70% | 963 |
| visual reuse + exact incremental prefill | 68% | 489 |
| full stack (adds one-step distillation) | 46% | lower, at real accuracy cost |

The middle row is the deployable one: half the energy, success statistically
indistinguishable from baseline (z=0.31, n=100 per condition), no training required,
about 30 lines of runtime logic.

Also established here:
- the failure boundary of visual reuse (success collapses 68% → 4% between 72% and 93% skip rates)
- an interaction: one-step distilled policies degrade far more than their ten-step teachers under identical visual staleness
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

## Method

SmolVLA (450M) decomposed into separately built TensorRT engines with an exact,
stateless per-step KV-cache contract. Engine outputs agree with fp32 reference
execution to a maximum action-space deviation of 2.5e-3. Power is sampled from the
module's INA3221 instrumentation at 10 ms intervals during replay of recorded episode
observations; task success is evaluated in the standard LIBERO simulator.

## Layout

- `paper/` — the paper (PDF and source), figures
- `scripts/` — export pipeline, measurement harness, evaluation
  - `export_engines.py`, `engines.py` — TensorRT decomposition
  - `edge_measure.py`, `edge_replay.py` — power sampling and replay
  - `eval_stack.py`, `eval_with_reuse.py`, `eval_onestep.py` — evaluation
  - `distill_lora.py` — one-step action-head distillation
  - `dump_reference.py`, `probe_onnx_export.py` — numerical validation against fp32

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
