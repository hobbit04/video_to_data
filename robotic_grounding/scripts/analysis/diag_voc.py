"""Instrumented zero-action rollout: which termination fires at VOC=1.0, and why."""
import argparse, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--task", type=str, default="Sharpa-V2D-v0")
p.add_argument("--motion_file", type=str, required=True)
p.add_argument("--voc", type=float, default=1.0)
p.add_argument("--steps", type=int, default=700)
p.add_argument("--first_frame", action="store_true",
               help="Reset every env to frame 0 so all envs share a phase, making the "
                    "per-step deviation comparable against the reference penetration curve.")
p.add_argument("--no_hand_object_collisions", action="store_true",
               help="Disable robot-to-object collisions, isolating the virtual controller's "
                    "own tracking error from whatever the hands do to the object.")
p.add_argument("--voc_stiffness", type=float, default=None,
               help="Override the virtual object controller's linear stiffness (default 50 N/m).")
p.add_argument("--voc_damping", type=float, default=None,
               help="Override its linear damping (default 10 N.s/m). d/k is the lag time constant.")
p.add_argument("--disable_away", action="store_true",
               help="Disable object_away_from_trajectory entirely and report the per-env "
                    "deviation distribution, i.e. the environment's own noise floor.")
p.add_argument("--derive_thresholds", action="store_true",
               help="Set object_away_from_trajectory thresholds to None so they are derived "
                    "from the reference motion (A-2), instead of the fixed 0.2 m / 0.7 rad.")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import torch, gymnasium as gym
import isaaclab_tasks  # noqa
from robotic_grounding.tasks import *  # noqa
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab.utils.math as math_utils

cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs, use_fabric=True)
cfg.motion_file = a.motion_file
apply_scene_config(cfg, SceneConfig.from_motion_file(cfg.motion_file))
cfg.viewer.env_index = 0
cfg.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale = a.voc
for _name, _term in vars(cfg.actions).items():
    if "virtual_" in _name and hasattr(_term, "tracking_controller_linear_stiffness"):
        if a.voc_stiffness is not None:
            _term.tracking_controller_linear_stiffness = a.voc_stiffness
        if a.voc_damping is not None:
            _term.tracking_controller_linear_damping = a.voc_damping
        print(f"[DIAG] {_name}: k={_term.tracking_controller_linear_stiffness} "
              f"d={_term.tracking_controller_linear_damping} "
              f"(lag time constant d/k = {_term.tracking_controller_linear_damping/_term.tracking_controller_linear_stiffness:.3f} s)", flush=True)
if a.first_frame:
    cfg.commands.dual_hands_object_tracking_command.always_reset_to_first_frame = True
if a.no_hand_object_collisions and hasattr(cfg.events, "setup_collision_groups"):
    cfg.events.setup_collision_groups.params["disable_robot_to_object_collisions"] = True
    print("[DIAG] robot-to-object collisions DISABLED", flush=True)
if a.disable_away:
    cfg.terminations.object_away_from_trajectory = None
if a.derive_thresholds:
    _p = cfg.terminations.object_away_from_trajectory.params
    _p["position_threshold"] = None
    _p["orientation_threshold"] = None
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
cmd = env.command_manager.get_term("dual_hands_object_tracking_command")
print(f"[DIAG] retargeted_horizon = {cmd.retargeted_horizon}", flush=True)
print(f"[DIAG] object mass = {env.scene['tissue_box_refined'].root_physx_view.get_masses()[0].tolist()}", flush=True)
print(f"[DIAG] reference travel {cmd.reference_object_travel:.4f} m, rotation {cmd.reference_object_rotation:.4f} rad", flush=True)
if "object_away_from_trajectory" in env.termination_manager.active_terms:
    _tp = env.termination_manager.get_term_cfg("object_away_from_trajectory").params
    print(f"[DIAG] thresholds in cfg: {_tp}", flush=True)
else:
    print("[DIAG] object_away_from_trajectory DISABLED -- measuring the noise floor", flush=True)

act = torch.zeros(env.action_space.shape, device=env.device)
tm = env.termination_manager
n_pos = n_ori = n_both = 0
counts = {k: 0 for k in tm.active_terms}
pos_hi = []; ori_hi = []; pos_all = []; ori_all = []; age_all = []; pos_mean = []; lift_rec = {}
for step in range(a.steps):
    with torch.inference_mode():
        env.step(act)
    dpos = torch.norm(cmd.object_body_position_command_e - cmd.object_position_e, dim=-1).max(dim=-1).values
    dori = math_utils.quat_error_magnitude(cmd.object_orientation_e, cmd.object_body_wxyz_command_e)
    if dori.dim() > 1: dori = dori.max(dim=-1).values
    pos_hi.append(dpos.max().item()); ori_hi.append(dori.max().item()); pos_mean.append(dpos.mean().item())
    for _k in ("object_lift_reference", "object_lift_achieved", "object_lift_ratio"):
        lift_rec.setdefault(_k, []).append(float(cmd.metrics[_k].mean().item()))
    pos_all.append(dpos.cpu()); ori_all.append(dori.cpu())
    age_all.append(cmd.steps_since_last_reset.flatten().cpu().clone())
    for k in tm.active_terms:
        counts[k] += int(tm.get_term(k).sum().item())
    oa = tm.get_term("object_away_from_trajectory") if "object_away_from_trajectory" in tm.active_terms else None
    if oa is not None and oa.any():
        m = oa
        pv = (dpos[m] > 0.2); ov = (dori[m] > 0.7)
        n_pos += int((pv & ~ov).sum()); n_ori += int((ov & ~pv).sum()); n_both += int((pv & ov).sum())
    if (step + 1) % 100 == 0:
        print(f"[DIAG] step {step+1}  counts={counts}  objAway(pos/ori/both)={n_pos}/{n_ori}/{n_both}"
              f"  max dpos={max(pos_hi):.3f}m  max dori={max(ori_hi):.3f}rad", flush=True)
print("[DIAG] FINAL", counts, "objAway pos-only/ori-only/both =", n_pos, n_ori, n_both, flush=True)
import numpy as np
print(f"[DIAG] dpos p50/p95/p99/max = {np.percentile(pos_hi,50):.4f}/{np.percentile(pos_hi,95):.4f}/{np.percentile(pos_hi,99):.4f}/{max(pos_hi):.4f} m", flush=True)
np.save("out/diag_voc_pos_mean.npy", np.array(pos_mean))
if lift_rec:
    _lr = np.array(lift_rec["object_lift_reference"]); _la = np.array(lift_rec["object_lift_achieved"])
    print(f"[DIAG] LIFT reference peak {_lr.max():.4f} m   achieved peak {_la.max():.4f} m", flush=True)
    _i = int(_lr.argmax())
    print(f"[DIAG] LIFT ratio at peak reference = {np.array(lift_rec['object_lift_ratio'])[_i]:.4f}", flush=True)
print(f"[DIAG] dori p50/p95/p99/max = {np.percentile(ori_hi,50):.4f}/{np.percentile(ori_hi,95):.4f}/{np.percentile(ori_hi,99):.4f}/{max(ori_hi):.4f} rad", flush=True)
pa = torch.cat(pos_all).numpy(); oa_ = torch.cat(ori_all).numpy()
print(f"[DIAG] PER-ENV dpos p50/p90/p99/p99.9/max = "
      f"{np.percentile(pa,50):.4f}/{np.percentile(pa,90):.4f}/{np.percentile(pa,99):.4f}/{np.percentile(pa,99.9):.4f}/{pa.max():.4f} m  (n={pa.size})", flush=True)
print(f"[DIAG] PER-ENV dori p50/p90/p99/p99.9/max = "
      f"{np.percentile(oa_,50):.4f}/{np.percentile(oa_,90):.4f}/{np.percentile(oa_,99):.4f}/{np.percentile(oa_,99.9):.4f}/{oa_.max():.4f} rad", flush=True)
for thr in (0.05, 0.078, 0.10, 0.12, 0.15, 0.20):
    print(f"[DIAG]   threshold {thr:.3f} m -> {(pa > thr).mean()*100:6.2f}% of env-steps exceed", flush=True)

# Split by how long ago the env was reset: a reset-ejection signature decays with
# age, a soft-controller signature does not.
age = torch.cat(age_all).numpy()
print("[DIAG] deviation vs steps-since-reset (VOC decay_steps = 20):", flush=True)
print(f"[DIAG]   {'age':>12} {'n':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}  {'>0.078m':>8}", flush=True)
for lo, hi in [(0, 5), (5, 20), (20, 40), (40, 80), (80, 160), (160, 10**9)]:
    m = (age >= lo) & (age < hi)
    if not m.any():
        continue
    v = pa[m]
    label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(f"[DIAG]   {label:>12} {v.size:>8} {np.percentile(v,50):8.4f} {np.percentile(v,90):8.4f} "
          f"{np.percentile(v,99):8.4f} {v.max():8.4f}  {(v>0.078).mean()*100:7.2f}%", flush=True)
env.close()
app.close()
