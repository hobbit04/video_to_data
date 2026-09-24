# Why the tissue-box run trained for 8 hours and learned nothing

An 8-hour PPO run (4079 iterations, 4096 envs, 401 M env steps) on a monocular
ego-video sequence produced a policy that raises the object **8.9 mm** where the
reference raises it **154 mm** — a lift ratio of **0.058**. Every metric in the
training log looked healthy: 95.8% of episodes ran to the end of the reference,
object position error fell from 11.3 cm to 3.7 cm, and mean reward rose from 50
to 354.

None of those numbers could distinguish success from failure on this clip. What
follows is the list of things in the default implementation that caused that,
each with the measurement that establishes it.

The root cause is not a bug. It is that several constants in the environment are
calibrated for mocap sequences where the object travels tens of centimetres,
while this clip's entire motion is 15.4 cm.

---

## 0. The result that reframes everything: zero actions beat every trained policy

Measured late, after A-1, B-1, C-1 and the contact-wrench run. The action space
is a **residual** on top of reference tracking, so a zero action means "track the
demonstration exactly". Replaying that with the virtual controller fully off:

| | lift ratio, VOC = 0, from frame 0 |
|---|---|
| **zero actions, de-penetrated reference** | **0.664** (0.1045 m of 0.1537 m) |
| zero actions, original reference | 0.590 |
| trained: 8 h, force_closure | 0.058 |
| trained: 1500, force_closure, after A-1 + B-1 | 0.057 |
| trained: 1500, contact wrench guidance | 0.051 |

**The demonstration lifts the box unaided. Every policy trained on it is more
than ten times worse than doing nothing.** The environment, the assets, the
friction and the retargeted grasp are all adequate; a solution sits at the
origin of the action space, and PPO walks away from it.

That inverts the reading of everything below. Sections A-D explain why the
reward does not *reward* lifting; the harder question this raises is why it
rewards something that actively *breaks* a grasp that already works.

It also corrects A-1's framing. The "do-nothing" baseline used there is a box
frozen at frame 0 scoring 0.7851, but that is not what a zero-action policy
does -- it lifts. With a p50 object deviation of 2.78 cm under zero actions,
`exp(-0.0278^2 / 0.0243)` puts the real zero-action keypoint reward near 0.97,
against the 0.757 the trained policy reaches. **Zero actions score higher on the
objective than the policy PPO converged to.** Widening the reward band was not
wrong, but it was never the binding constraint.

### The curriculum is not what breaks it, and the objective is

Re-run with VOC at 0 from iteration 0, everything else held fixed against the
contact-wrench run. The lift starts near the zero-action level and is optimised
away:

| iteration | 4 | 100 | 200 | 300 | 500 | 1000 | 1499 |
|---|---|---|---|---|---|---|---|
| `object_lift_ratio` | **0.195** | 0.089 | 0.064 | 0.033 | 0.022 | 0.026 | **0.024** |
| `objAway` | 0.275 | 0.563 | 0.546 | 0.469 | 0.255 | 0.229 | **0.109** |
| mean reward | 7.5 | 17.8 | 24.8 | 33.0 | 42.7 | 90.3 | **262.1** |
| object error | — | 6.87 | 8.58 cm | 6.80 | 4.88 | 5.29 | **3.74 cm** |

Final policy: lift ratio **0.044** from frame 0, the worst of the four runs.

**PPO improves its objective monotonically while destroying the lift
monotonically.** Over the run, `object_lift_ratio` correlates with mean reward
at **-0.78** (Spearman) and with `objAway` at **+0.83**. Nothing here is a
curriculum artefact; there was no curriculum.

The `objAway` correlation is the mechanism. The termination meant to catch
failure instead teaches the policy to keep the object still: lifting a 0.3 kg box
with an imperfect grasp is the single most likely way to displace it past the
threshold, and a termination costs the whole remaining reward stream. The policy
drove `objAway` from 0.56 to 0.11 over the run, and the lift went with it. It
did not fail to learn -- it learned that holding the box down is safe.

Two separate losses, both measured:

| | lift ratio | lost to |
|---|---|---|
| zero actions | 0.238 | — |
| policy at initialisation | ~0.092 | `init_noise_std` 0.1 perturbing a working grasp (-61%) |
| policy after training | 0.019 | the objective (-79%) |

So `init_noise_std` costs more than half the lift before a single gradient step,
and optimisation removes most of what is left. A fix has to address both: the
exploration scale relative to what a grasp tolerates, and an objective under
which moving the object is not the risky choice.

The next experiments, in order of how much they would tell us:

1. **Behaviour cloning / residual regularisation toward zero.** The
   demonstration already solves the task. Penalise the residual, or initialise
   and anchor the policy at zero action, so optimisation has to earn its
   departure from a working solution.
2. **Drop or invert `object_away_from_trajectory`.** It is currently a
   don't-touch-the-object incentive. A lift-based termination (the object failed
   to leave the table by the frame the reference does) inverts the sign.
3. **Shrink `init_noise_std` and the residual scales.** 0.1 on 56 dimensions with
   5 cm wrist and 0.15 rad finger scales is large next to the tolerance of a
   two-handed grasp on a 20 cm box.

Measured by `diag_voc.py --voc 0.0 --first_frame --disable_away` and
`../rsl_rl/diag_policy.py`.

---

## A. Absolute constants fixed at mocap scale — primary cause

### A-1. `object_keypoints_tracking_exp.var = 0.1`

**Default** — `tasks/v2d/v2d_hand_env_cfg.py:219-226` hardcodes `"var": 0.1`.
The reward is `exp(-||Δkeypoint||² / var)`, so this declares an error resolution
of `sqrt(0.1) ≈ 0.32 m`.

**Why it is wrong here** — the object's entire travel is 0.154 m, so the
reward is coarser than the task. A policy that freezes the object at frame 0
scores **0.9326** where perfect tracking scores 1.0, leaving a **6.7% reward
band for the whole task**. In return terms that is 22 points, against the
~147 points of remaining reward stream forfeited by an early termination.
Standing still is the rational choice under this objective.

**Fix — IMPLEMENTED.** `var` now accepts `None`, in which case
`rewards.py:_derive_keypoint_var` computes it from the reference motion's own
extent as `var = travel²`, clamped to `[0.005, 0.1]`. That reproduces the 0.1
default for a ~0.32 m mocap motion and sharpens automatically for short clips.
The command term measures the extent once at load
(`hand_object_commands.py:_init_reference_motion_scale`) and exposes it as
`reference_object_travel` / `reference_object_rotation`, both computed as
trajectory diameters so they stay valid under the random-frame resets.

The shipped default stays at 0.1 — deriving is opt-in per run, so no other
dataset changes behaviour:

```
env.rewards.object_keypoints_tracking_exp.params.var=null
```

| var | do-nothing score | task reward band |
|---|---|---|
| 0.1 (default) | 0.9326 | 6.7% |
| **derived = 0.0243** | 0.7851 | **21.5%** |
| 0.01 (hand-picked, for reference) | 0.6377 | 36.2% |

The derivation lands on 0.0243 for this clip, within a hair of the 0.02 that
was arrived at by hand. `train_tissue_box_refined_8h.sh` now passes `var=null`.

Two lines confirm it. `[v2d] reference object travel 0.1559 m, rotation
0.0963 rad` prints at startup when the motion loads. `[v2d]
object_keypoints_tracking_exp: derived var=0.02432` prints only at the first
curriculum step that gives the term a non-zero weight (iteration 526 in the
compressed schedule), because Isaac Lab's `RewardManager.compute` skips terms
whose weight is 0 and so never calls the function before then. Both were
verified end to end with a 2-iteration smoke run.

Measured by `rew_headroom.py`, `rew_sweep.py`, `verify_A.py`.

### A-2. `object_away_from_trajectory` thresholds 0.2 m / 0.7 rad

**Default** — `tasks/v2d/v2d_hand_env_cfg.py:311-318`. The term compares the
object's actual pose against what the reference prescribes *at the current
timestep*.

**Why it is wrong here** — for a policy that leaves the object on the table,
the error it feeds the term is just the reference's own displacement:

```
position  max 0.1555 m   (threshold 0.20 m, 22% margin)
rotation  max 0.0965 rad (threshold 0.70 rad, 86% margin)
```

**Neither condition can ever fire.** Never lifting the object is, by this
criterion, following the trajectory. That is why `time_out` reached 95.8% and
was misread as a success rate.

**Mechanism IMPLEMENTED, not yet enabled — it needs A-2b below first.**
`position_threshold` / `orientation_threshold` now accept `None`, deriving
`0.5 × reference extent` with floors of 0.05 m / 0.2 rad
(`terminations.py:_derive_away_thresholds`). For this clip that gives
**0.078 m / 0.200 rad**, and a frozen object would terminate at frame 140 —
exactly the discrimination the term is supposed to provide.

It could not be turned on as shipped. With the term disabled and **zero actions
at VOC = 1.0** — the object driven by the virtual controller along the
reference, nothing else happening — the per-env deviation over 358,400
env-steps is:

```
p50 0.0313   p90 0.0906   p99 0.1095   p99.9 0.1346   max 0.2508 m
```

| candidate threshold | env-steps exceeding it under zero actions |
|---|---|
| 0.050 m | 36.4% |
| **0.078 m (derived)** | **19.2%** |
| 0.100 m | 5.4% |
| 0.120 m | 0.3% |
| 0.200 m (shipped) | 0.0% |

Enabling the derived threshold terminated **62% of episodes** (2202 objAway
against 1337 timeouts) with the policy doing nothing at all. The environment's
own tracking noise was nearly as large as the reference's entire motion, so no
threshold both discriminated and stayed off the noise.

**The cause is the hands pushing the object, driven by this penetration.**
Three measurements pin it down.

First, the virtual controller is not at fault. Disabling robot-to-object
collisions leaves it tracking the reference alone:

| | collisions on (shipped) | collisions off |
|---|---|---|
| p50 | 3.13 cm | **0.26 cm** |
| p90 | 9.06 cm | **0.74 cm** |
| p99 | 10.95 cm | **0.82 cm** |
| max | 25.08 cm | **0.84 cm** |
| exceeding 0.078 m | 19.2% | **0.00%** |

That 0.84 cm ceiling is exactly what the controller's own lag predicts. The
wrench is `50·Δp − 10·v` with no feed-forward of the reference's velocity, so
holding velocity `v` requires `(d/k)·v` of position error, and accelerating at
`a` requires `(m/k)·a`. On the env timeline (`motion_speed 0.5` stretches the
13.0 s clip to 25.9 s, halving all speeds) the reference object peaks at
0.039 m/s and 0.23 m/s², giving a budget of **0.83 cm max** against the 0.84 cm
measured. The controller does its job to within a centimetre; **97% of the
deviation is the hands.**

Second, the deviation tracks the penetration. Running every env in phase and
comparing the per-step deviation against the reference's per-frame penetration
curve gives Pearson r = **+0.56** (Spearman +0.54), and the deviation roughly
doubles between the low- and high-penetration halves of the trajectory
(2.86 cm against 5.60 cm).

Third, the force budget explains why the hands win. Each hand's wrench is capped
at 60 N per axis, while the object controller produces only `50 × 0.03 = 1.5 N`
at a 3 cm error. The hands overpower it by more than an order of magnitude.

So the chain is: the retargeted reference commands the hands 2.19 cm inside the
box → the contact solver applies separating impulses for as long as contact
lasts → the object is shoved 3–11 cm off the reference → that noise floor
overlaps the failure signal (a non-lifting policy reaches 13.9 cm) → no
threshold separates them. **B-1 is the fix.**

Stiffening the controller also works, but by masking rather than curing — at
`k=500` a 3 cm error produces 15 N, enough to out-muscle the hands:

| gains | p50 | p90 | p99 | exceeding 0.078 m |
|---|---|---|---|---|
| k=50, d=10 (shipped) | 0.0313 | 0.0906 | 0.1095 | **19.2%** |
| **k=500, d=30** | **0.0044** | **0.0113** | **0.0280** | **0.02%** |

Both gain sets are stable (at k=500 on 0.3 kg, `omega_n = 40.8 rad/s`,
`zeta = 1.22`, `omega_n*dt = 0.41` at the 100 Hz inner loop). It is a legitimate
lever if a sharper signal is needed before the reference can be re-fitted, but
it leaves the hands crushing the object and only hides it, and a stiffer aid
makes the assisted phase more idealised than the unassisted one the curriculum
decays to. Prefer fixing B-1.

Note the orientation channel is separately under-damped — `K=10`, `D=0.1`,
`I=0.001` gives `zeta = 0.5`, and the zero-action run reaches 2.4 rad of
ringing. The derived 0.2 rad floor keeps that from terminating episodes, but it
is worth fixing on its own.

Recommended order: reduce the penetration (B-1), re-run
`diag_voc.py --disable_away` to confirm the floor has dropped, then enable
`position_threshold=null`.

Recommended order: retune the controller, re-run `diag_voc.py --disable_away`
to confirm the gap, then enable `position_threshold=null`.

Note the orientation channel is separately under-damped — `K=10`, `D=0.1`,
`I=0.001` gives `zeta = 0.5`, and the zero-action run reaches 2.4 rad of
ringing. The derived 0.2 rad floor keeps that from terminating episodes, but it
is worth fixing on its own.


Measured by `objaway_margin.py`, `lift_timing.py`, `diag_voc.py`.

### A-3. `KEYPOINT_VECS` — verified, do not touch (documented in code)

`tasks/v2d/mdp/commands/hand_object_commands.py:290-304` places the six object
keypoints at **1 m** from the object centre regardless of object size. This
looks like the obvious culprit for a 20 cm box, but shrinking the lever makes
the reward band **worse**, not better (6.7% → 5.4% at 0.2 m). Most of the 6.7%
comes from the small orientation change being amplified by the long lever;
remove it and only the position error is left, which discriminates even less.
Do not spend effort here. This is now recorded in the
`object_keypoints_tracking_exp` docstring so the next reader does not repeat it.

Measured by `rew_sweep.py`.

---

## B. Reference data does not pass the paper's quality gate — secondary cause

### B-1. Retargeted sequence penetrates the object by 2.19 cm

**Default** — the paper filters out sequences whose hand-object penetration
exceeds 2 cm before training. `scripts/data_quality_checks/` implements this,
but the pipeline does not enforce it ahead of training.

**Why it matters** — the sequence used for the 8-hour run scores **2.188 cm**
and would have been rejected (left hand, around frame 269; right hand peaks at
1.48 cm). This propagates: every reset teleports the hands into a pose that is
inside the box, the contact solver ejects them, and **`object_away` fires 0.36%
of the time even with zero actions**, with deviations reaching 0.197 m — 98% of
the termination threshold.

**Gate IMPLEMENTED.** `train_tissue_box_refined_8h.sh` now runs
`data_assessor.py --reject` before launching and aborts on failure;
`data_assessor` already exits non-zero under `--reject`, so only the wiring was
missing. It passes `stride=1`, because the default `stride=3` samples every
third frame and can step over the deepest one. `SKIP_PREFLIGHT=1` overrides it
deliberately.

**FIXED, with a flag that already existed.** `ego_recon_to_sharpa.py` ships
`--surface_project`, which pushes MANO joint IK targets out of the object's OBB
proxy before IK; the original retarget simply did not use it. No new penetration
penalty was needed, and **reconstruction did not have to be re-run** — the
loader output is cached and de-penetration happens at the retarget stage.

| `--surface_margin` | max | mean | p90 | >2cm | >1cm | IK task error R/L |
|---|---|---|---|---|---|---|
| none (original) | **2.19** | 1.13 | 1.76 | **19/390** | 270/390 | 2.28 / 2.32 cm |
| 0.000 | 1.48 | 0.89 | 1.39 | 0/390 | 207/390 | 2.34 / 2.40 cm |
| **0.005 (default)** | **1.39** | 0.74 | 1.10 | **0/390** | 105/390 | 2.38 / 2.49 cm |
| 0.010 | 1.36 | 0.67 | 1.04 | 0/390 | 57/390 | 2.45 / 2.57 cm |
| 0.020 | 1.36 | 0.60 | 0.98 | 0/390 | 30/390 | 2.65 / 2.80 cm |

The max saturates at 1.36 cm past 0.005 while the IK error keeps climbing, so
0.005 — the shipped default — is the right setting. Contact fraction is
unchanged at 77.2%. Both sequences were re-retargeted with it and the support
surfaces regenerated; `tissue_box_refined` now scores **1.392 cm and passes**
both gates (penetration, and the 300-step replay). `tissue_box_timing`, the
un-refined control, improves 2.86 → 2.02 cm and still fails at the same margin
— a measure of what the gsplat refinement is worth.

**It does NOT unblock A-2, contrary to what this section previously claimed.**
Re-measuring the zero-action noise floor on the de-penetrated sequence:

| | penetration 2.19 cm | penetration 1.39 cm |
|---|---|---|
| p50 | 0.0313 | 0.0265 |
| p90 | 0.0906 | 0.0968 |
| p99 | 0.1095 | 0.1137 |
| exceeding 0.078 m | 19.2% | **19.8%** |

Unchanged. The hands are the cause — that much is established by intervention,
since disabling hand-object collisions drops the deviation to 0.26 cm — but the
penetration *depth* is not the knob that sets how hard they push. A grasp in
contact at all exerts forces far above the 1.5 N the object controller produces
at a 3 cm error, and the hands are position-controlled to reference wrist poses
with no notion of holding gently. The only lever with an intervention behind it
remains the controller gains (A-2).

**Correction — the reset-ejection story was wrong, but the penetration is still
on A-2's critical path.** Splitting the zero-action deviation by
steps-since-reset shows the reset is the *calmest* part of an episode, not the
worst:

| steps since reset | p50 | p90 | exceeding 0.078 m |
|---|---|---|---|
| 0–5 | 0.0081 | 0.0509 | **0.30%** |
| 5–20 | 0.0161 | 0.0747 | 8.19% |
| 20–40 | 0.0177 | 0.0977 | 20.08% |
| 40–80 | 0.0279 | 0.1016 | 26.04% |
| 80–160 | 0.0469 | 0.0997 | **29.14%** |
| 160+ | 0.0285 | 0.0814 | 13.20% |

The reset teleports the object exactly onto the reference, so the deviation
starts at zero by construction and the VOC is pinned at 1.0 with the reference
clock stopped for the first 20 steps. What the growth with age actually shows is
the hands slowly shoving the object out of place once contact is live — a
*continuous* mechanism, not an impulse at reset. A-2 has the measurements.

Measured by `pen_per_hand.py`, `diag_voc.py`.

### B-2. `hull_volume_ratio` gate misclassifies non-watertight meshes, and reports the skip as a pass

**Default** — `scripts/filter_penetrations.py:752` sets `hull_ratio_max = 3.0`.
When `hull.volume / mesh.volume` exceeds it, the object is treated as concave
(AR glasses, mugs, open vases) and the hand-object check is skipped entirely.

**Why it is wrong here** — the SAM3D reconstructed mesh is not watertight, so
`trimesh` computes a volume about 1/6 of the true one (0.000367 vs
0.00234 m³). The ratio comes out at 6.284 and **a convex rectangular tissue box
is classified as concave**. The gate assumes the clean scanned meshes of a mocap
dataset; a monocular reconstruction does not satisfy that assumption.

**Worse** — the skipped check is reported as `{"pass": True, "score": 0.0}`.
The `"no frames"` path at `filter_penetrations.py:770` does the same. **"Not
measured" is indistinguishable from "passed"**; the first run of this check
returned `pass 100%, score 0.0000` and was nearly taken at face value.

**Fix — IMPLEMENTED.** Two changes, neither of which needs the mesh repaired.

`_load_hull` now computes the ratio **only when the mesh is watertight** and
returns `None` otherwise, instead of fabricating one from a meaningless volume.
A `None` ratio no longer skips the check: it runs against the convex hull and
the result is flagged `[hull proxy unverified (non-watertight)]`, so a reader
knows the depth could be pessimistic for a genuinely concave object. Repairing
the mesh was tried and rejected — `trimesh.repair.fill_holes` does not close
this mesh, and a voxel-fill volume is pitch dependent (0.00178 m³ at a 0.65 cm
pitch against 0.00111 m³ at 0.32 cm), so it would have swapped one arbitrary
number for another.

`check()` can no longer return a pass for something it did not measure. When
every body is skipped, or the sequence has no frames, it returns
`{"pass": False, "score": NaN, "reason": "NOT MEASURED: ..."}`; a partial skip
appends the reason and fails. `data_assessor` counts NaN scores in a separate
**Not measured** column instead of averaging them in as zeros.

The same command that used to report a clean sweep now reports the truth:

| | before | after |
|---|---|---|
| `tissue_box_refined` | `pass, 0.0, "ok (max=0.00cm)"` | `FAIL, 2.18 cm, hull proxy unverified` |
| `tissue_box_timing` | `pass, 0.0, "ok (max=0.00cm)"` | `FAIL, 2.56 cm, hull proxy unverified` |
| `tissue_box_simple` | `pass, 0.0, "no frames"` | `FAIL, NaN, "NOT MEASURED: no frames"` |
| summary | `Pass Rate 100%, Mean Score 0.0000` | `Pass Rate 0.0%, Mean 2.3690, Not measured 1` |

No `hull_ratio_max` override is needed any more. Saved as
`qc_penetration_default_after_B2.json` beside the pre-fix `qc_penetration.json`.

---

## C. Nothing observes success

### C-1. No lift metric

**Default** — the environment has no termination term and no metric that asks
whether the object was lifted. The log carries tracking errors and termination
rates only.

**Why it matters** — there was no way to know the run had failed until the
video was watched eight hours later. The number that makes it obvious,
lift ratio 0.058, only exists because a diagnostic script was written after the
fact.

**Fix — IMPLEMENTED** as a **metric, not a reward**, so the objective stays the
paper's. `hand_object_commands.py:_update_lift_metrics` logs three values:

| metric | meaning |
|---|---|
| `object_lift_reference` | running max of the reference's rise above the object's height at reset (m) |
| `object_lift_achieved` | the same for the actual object (m) |
| `object_lift_ratio` | achieved / reference, reported as 0 until the reference has lifted at least 1 cm |

Both lifts share one baseline, the object's height at reset, so they stay
meaningful under the random-frame resets: an episode starting mid-air is scored
on the lift remaining from there, not on the whole trajectory. Running maxima
rather than instantaneous heights, so an episode that lifts and sets down again
still reads as a lift. Isaac Lab reads command metrics at reset, so what lands
in the log is the end-of-episode value.

Checked against the failed policy, where the offline computation gives a lift
ratio of 0.053-0.058: the env-side metric reports `object_lift_reference`
0.1537 m against `object_lift_achieved` 0.0105 m, ratio 0.0656 mid-episode. A
40-iteration training run confirms the three lines reach the console log.

### C-2. `Episode_Termination/time_out` reads as a success rate

**Default** — Isaac Lab's `TerminationManager.reset()` logs
`_last_episode_dones.float().mean(dim=0)`: the fraction of **environments**
whose most recent episode ended with each cause. The statistic itself is
correct and, being per-environment rather than per-episode, is not biased by
short failing episodes resetting more often.

**Why it misleads** — in an environment with no success criterion, `time_out`
is the only number that looks like things are working, so a reader takes it for
a success rate. Its actual meaning is "was not terminated early".

**Fix** — C-1 resolves this. Separately, do not label it "completion rate" in
reports.

---

## D. Training configuration

### D-1. `wrist_position_clip` equals the wrist termination threshold

`tasks/v2d/mdp/actions/actions_cfg.py:48` sets `wrist_position_clip = 0.2`, and
`v2d_hand_env_cfg.py:303-309` sets
`hand_wrist_away_from_trajectory.threshold = 0.2`. The policy's maximum
commandable wrist residual is exactly the termination boundary, and it used it:
the final right-wrist error sat at **0.171 m**, just inside. Keep the clip
below the threshold (e.g. 0.12) so that "died from its own command" and "died
from a physics accident" stay distinguishable.

### D-2. `force_closure_reward` has no per-hand floor

`tasks/v2d/mdp/rewards.py:627-676` averages over reference-active hands, so
abandoning one hand still collects half the reward. The run split: right-hand
wrench support **0.493**, left **0.743**, with the right wrist hovering 4.5 cm
further from the box than it should.

This is emergent symmetry breaking, not a structural bias. Ruled out:
right/left robot configs are identical except the URDF path; the reference is
symmetric (wrist-object distance 0.209 vs 0.210 m, path length 0.604 vs
0.639 m, contact frames 218 vs 230); and the **left** hand is the one with
deeper penetration (2.19 vs 1.48 cm), so that cannot explain a worse right
hand. The asymmetry also flips mid-training — at iteration 1000 the left finger
error was 1.035 against the right's 0.220.

**Fix** — require a minimum per-hand support, or combine the hands
multiplicatively (`sqrt(left × right)`). This changes a reward function, so
defer it behind A-1 and B-1, and first confirm the symmetry break reproduces
under a different seed.

### D-3. `reset_to_first_frame_prob` is gated on VOC — low priority, NOT the cause

`hand_object_commands.py:1907-1920` only applies the frame-0 reset once
`virtual_object_controller_scale_factor < 0.1`, which in this run meant from
iteration 2105, giving 4.8% frame-0 exposure overall against an eval that
always starts at frame 0.

**This was initially blamed and then refuted.** Replaying the same checkpoint
from random start frames:

| | start at frame 0 | random start frame |
|---|---|---|
| mean z(actual) − z(reference) | −4.47 cm | −4.92 cm |
| steps more than 3 cm below reference | 42.9% | **99.4%** |

71% of training episodes began with the object already in the air, and the
policy still failed to hold it there. Failure is uniform across the state
distribution, which points at the objective, not at coverage. Ungating this is
a cheap and harmless change, but on its own it changes nothing.

Measured by `../rsl_rl/diag_policy.py --random_start`.

### D-4. Curriculum compression — secondary

The schedule was compressed by 0.2632 to fit 4079 iterations, leaving **395**
iterations after VOC reached 0 against **6000** in the stock schedule. But the
weight-normalised rewards plateau at iteration ~900 and stay flat within each
curriculum stage, so there is little evidence more iterations would have helped.
Re-evaluate after A and B.

### D-5. Do not raise the iteration budget

The stock schedule implies 20,000 iterations (≈39 h here) and there is no
evidence for it. `force_closure` peaks at 0.690 around iteration 900 and drifts
down to 0.650 over the following 3,100 iterations; the hand tracking terms start
at 0.985 and only decline. Fix A-1, B-1 and C-1, then run **1,000–1,500
iterations** and check whether the lift ratio moves at all.

To recover these numbers from the log, note that `Episode_Reward/<term>` is
`sum(func × weight × dt) / max_episode_length_s` with `dt = 0.05` and
`max_episode_length_s = 26.0`. Undo it with
`Episode_Reward × 26 / (weight × mean_episode_length × 0.05)` to get the raw
0..1 term value. `train_8h_metrics.csv` carries the raw columns.

---

## E. Repository bugs found along the way, unrelated to this experiment

| where | problem | fix |
|---|---|---|
| `scripts/rsl_rl/eval.py` | does not clamp `viewer.env_index = 6`, so `--num_envs < 7` always fails | port the two lines from `dummy_agent.py:204-207` |
| `reconstruction/modules/v2d_pipelines/run_ego_wilor.py` | `_step("Package result/", ...)` skips when `result/` exists, so **gsplat-refined poses never reach the bundle** | re-package when refinement ran |
| `reconstruction/modules/v2d_task_library_loader/lib/level_result_bundle.py` | axis evaluation and application each run the same 90k-rotation brute force (4 min × 2) | cache the search, or build `--up_rank` comparison in |

---

## Suggested order

1. ~~**B-2** (surface the skip reason)~~ — done.
2. ~~**A-1** (derive `var`)~~ — done; `var=null` in the training script.
3. ~~**B-1 gate**~~ — done; the training script aborts on a failing sequence.
4. ~~**B-1 proper**~~ — done; `--surface_project --surface_margin 0.005` takes
   `tissue_box_refined` to 1.392 cm and both gates pass. It did **not** unblock
   A-2: the noise floor is unchanged at 19.8% exceedance.
5. **A-2b: retune the virtual object controller** to `k=500, d=30`, or better,
   scale the gains with object mass. This is the only lever with an
   intervention behind it (19.2% -> 0.02% exceedance).
6. ~~**C-1** (log lift ratio)~~ — done; `object_lift_ratio` is in the training log.
7. ~~A 1,500-iteration run~~ — done, and the lift ratio did **not** move:
   0.058 before, **0.057** after (8.6 mm raised against the reference's 153 mm),
   with the config verified applied. `objAway` also rose to 0.16-0.59 against
   0.04-0.08 in the 8 h run, i.e. the policy displaces the object past the 0.2 m
   threshold more often without ever lifting it. Caveat: 1500 iterations is
   2.7x fewer than the 8 h run and the curriculum shape differs, so this bounds
   the effect of A-1 + B-1 rather than measuring it exactly.
8. ~~**Turn on `contact_wrench_support_reward`**~~ — done; lift ratio 0.051,
   and the wrench reward itself plateaus at 0.30 of its ceiling within ~100
   iterations and never moves again over the remaining 1400. It did fix the
   left/right asymmetry of D-2 (wrench support 0.488 / 0.488, against
   0.493 / 0.743 under force_closure) and put both wrists within 1 cm of their
   reference offset, so the grasp got better while the lift did not.
9. ~~**Run with VOC off from the start**~~ — done, and it refutes the curriculum
   hypothesis: lift ratio 0.044, the worst of the four runs, decaying
   monotonically from 0.195 while the reward rose from 7.5 to 262. See
   section 0 for what it established instead and what to try next.
10. **A-2** (enable `position_threshold=null`), then **D-1**, **D-2**, **D-3**.
11. **D-4/D-5** — re-evaluate the curriculum length only after the above.

Steps 1–4 are configuration changes and added observability only, so the run
remains a faithful CHORD reproduction. D-2 is the sole reward-function change
and it can wait.
