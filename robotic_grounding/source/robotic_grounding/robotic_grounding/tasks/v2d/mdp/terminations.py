# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hand_to_object_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when hands deviate too far from the commanded trajectory.

    Compares per-hand wrist-to-object distances against the commanded
    wrist-to-object distances and terminates if any hand exceeds
    `threshold` times its commanded distance.

    Args:
        env: The environment instance.
        command_name: The name of the command term.
        threshold: Ratio threshold for termination.

    Returns:
        Tensor of shape (num_envs,) indicating whether to terminate.
    """
    command = env.command_manager.get_term(command_name)

    right_hand_wrist_object_position_difference_command = torch.norm(
        command.right_hand_wrist_pose_command_e[:, :3]
        - command.object_body_position_command_e,
        dim=-1,
    )
    left_hand_wrist_object_position_difference_command = torch.norm(
        command.left_hand_wrist_pose_command_e[:, :3]
        - command.object_body_position_command_e,
        dim=-1,
    )

    right_hand_wrist_object_position_difference = torch.norm(
        command.right_robot.data.body_link_pos_w[:, command.right_wrist_body_id]
        - command.object_position_w,
        dim=-1,
    ).squeeze()
    left_hand_wrist_object_position_difference = torch.norm(
        command.left_robot.data.body_link_pos_w[:, command.left_wrist_body_id]
        - command.object_position_w,
        dim=-1,
    ).squeeze()

    right_hand_difference_ratio = (
        right_hand_wrist_object_position_difference
        / right_hand_wrist_object_position_difference_command
    )
    left_hand_difference_ratio = (
        left_hand_wrist_object_position_difference
        / left_hand_wrist_object_position_difference_command
    )

    return torch.logical_and(
        right_hand_difference_ratio > threshold,
        left_hand_difference_ratio > threshold,
    )


def hand_wrist_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when the hands are away from the trajectory."""
    command = env.command_manager.get_term(command_name)
    right_hand_position_difference = torch.norm(
        command.right_hand_wrist_pose_command_e[:, :3]
        - command.right_hand_wrist_position_e,
        dim=-1,
    )
    left_hand_position_difference = torch.norm(
        command.left_hand_wrist_pose_command_e[:, :3]
        - command.left_hand_wrist_position_e,
        dim=-1,
    )
    return torch.logical_or(
        right_hand_position_difference > threshold,
        left_hand_position_difference > threshold,
    )


def object_away_from_trajectory_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when the object is away from the trajectory.

    Args:
        env: The environment instance.
        command_name: The name of the command.
        threshold: The threshold for the termination.

    Returns:
        Tensor of shape (num_envs,) indicating whether to terminate.
    """
    command = env.command_manager.get_term(command_name)
    object_position_z_difference = torch.abs(
        command.object_body_position_command_e[..., 2]
        - command.object_position_e[..., 2].squeeze()
    )
    return object_position_z_difference > threshold


class ObjectLiftFailed(ManagerTermBase):
    """Terminate when the demonstration has lifted the object and the policy has not.

    ``object_away_from_trajectory`` asks how far the object is from the reference
    pose, which on a short clip cannot tell "left the object on the table" from
    "carried it with a few centimetres of hand-induced offset": the retargeted
    grasp itself pushes the object 3-10 cm off the reference while holding it,
    so any deviation budget tight enough to catch a non-lift also terminates
    the demonstration. This term asks the task's own question instead.

    Both lifts are measured above the object's height at reset, the same
    baseline as the ``object_lift_*`` metrics, so the term is well defined under
    random-frame resets and inert while the reference has not yet risen. The
    reference side is *lagged* by ``lag_steps``: measured on the tissue-box
    clip, the demonstration's own box starts rising 1-2 s after the reference
    does and only then catches up (zero actions, VOC off, from frame 0: 71% of
    envs are below 0.3 x reference the moment the reference passes 5 cm, 0% by
    the time it passes 15 cm). Comparing the object's running-max rise against
    the reference's running-max rise from ``lag_steps`` ago lets the
    demonstration pass while a policy that never lifts is still cut short well
    before the episode ends.

    Off by default (``enabled=False`` returns all-False) so the shipped
    configuration is unchanged; enable per run with
    ``env.terminations.object_lift_failed.params.enabled=true``.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._lag = int(cfg.params.get("lag_steps", 40))
        self._ref_hist = torch.zeros(env.num_envs, max(self._lag, 1), device=env.device)
        self._ptr = 0
        self._max_ref_lagged = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._ref_hist[env_ids] = 0.0
        self._max_ref_lagged[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        reference_lift_min: float = 0.05,
        achieved_lift_ratio_min: float = 0.3,
        lag_steps: int = 40,
        enabled: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            env: The environment instance.
            command_name: The name of the command term.
            reference_lift_min: The lagged reference must have risen at least
                this far (metres, running max since reset) before the check applies.
            achieved_lift_ratio_min: Terminate when the object's own running-max
                rise is below this fraction of the lagged reference's.
            lag_steps: How many env steps behind the reference the object is
                allowed to be. Read once at construction.
            enabled: Off by default so the shipped configuration is unchanged.

        Returns:
            Tensor of shape (num_envs,) indicating whether to terminate.
        """
        if not enabled:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        command = env.command_manager.get_term(command_name)
        reference_lift_now = (
            command.object_body_position_command_e[:, 0, 2] - command.reset_object_z
        )
        # Ring buffer: the slot being overwritten holds the value from lag_steps ago.
        lagged = self._ref_hist[:, self._ptr].clone()
        self._ref_hist[:, self._ptr] = reference_lift_now
        self._ptr = (self._ptr + 1) % self._ref_hist.shape[1]
        torch.maximum(self._max_ref_lagged, lagged, out=self._max_ref_lagged)
        return (self._max_ref_lagged >= reference_lift_min) & (
            command.max_achieved_lift < achieved_lift_ratio_min * self._max_ref_lagged
        )


#: Fraction of the reference motion's own extent used as the deviation budget
#: when ``object_away_from_trajectory`` derives its thresholds, plus the floors
#: that keep a near-static demonstration from terminating on solver noise.
_DERIVED_THRESHOLD_FRACTION = 0.5
_DERIVED_POSITION_FLOOR = 0.05  # metres
_DERIVED_ORIENTATION_FLOOR = 0.2  # radians


def _derive_away_thresholds(command) -> tuple[float, float]:
    """Scale the "object left the trajectory" budget to how far the reference moves.

    The term asks how far the object is from where the reference says it should
    be *now*, so for a policy that never moves the object the error it sees is
    just the reference's own displacement.  With fixed 0.2 m / 0.7 rad
    thresholds that makes the term vacuous on any clip whose motion is smaller
    than the threshold -- never lifting the object then reads as following the
    trajectory perfectly, and the episode runs to timeout.

    Deriving the budget as a fraction of the reference's extent keeps the term
    meaningful at any motion scale: it reproduces roughly the 0.2 m default for
    a ~0.4 m mocap motion, and tightens automatically for short clips.
    """
    cached = getattr(command, "_derived_away_thresholds", None)
    if cached is not None:
        return cached
    travel = getattr(command, "reference_object_travel", None)
    rotation = getattr(command, "reference_object_rotation", None)
    if travel is None or rotation is None:
        raise ValueError(
            "threshold=None requires the command term to expose "
            "reference_object_travel / reference_object_rotation; pass explicit "
            "thresholds for command terms that do not."
        )
    derived = (
        max(_DERIVED_THRESHOLD_FRACTION * travel, _DERIVED_POSITION_FLOOR),
        max(_DERIVED_THRESHOLD_FRACTION * rotation, _DERIVED_ORIENTATION_FLOOR),
    )
    print(
        f"[v2d] object_away_from_trajectory: derived thresholds "
        f"{derived[0]:.4f} m / {derived[1]:.4f} rad from reference extent "
        f"{travel:.4f} m / {rotation:.4f} rad (fixed defaults are 0.2 / 0.7)",
        flush=True,
    )
    command._derived_away_thresholds = derived
    return derived


def object_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float | None,
    orientation_threshold: float | None,
) -> torch.Tensor:
    """Terminate when the object is away from the trajectory.

    Args:
        env: The environment instance.
        command_name: The name of the command term.
        position_threshold: Deviation budget in metres, or ``None`` to derive it
            from the reference motion -- see :func:`_derive_away_thresholds`.
        orientation_threshold: The same, in radians.

    Returns:
        Tensor of shape (num_envs,) indicating whether to terminate.
    """
    command = env.command_manager.get_term(command_name)
    if position_threshold is None or orientation_threshold is None:
        derived_position, derived_orientation = _derive_away_thresholds(command)
        if position_threshold is None:
            position_threshold = derived_position
        if orientation_threshold is None:
            orientation_threshold = derived_orientation
    object_position_difference = torch.norm(
        command.object_body_position_command_e - command.object_position_e,
        dim=-1,
    )
    object_orientation_difference = math_utils.quat_error_magnitude(
        command.object_orientation_e,
        command.object_body_wxyz_command_e,
    )
    return torch.logical_or(
        (object_position_difference > position_threshold).any(dim=-1),
        (object_orientation_difference > orientation_threshold).any(dim=-1),
    )


def timestep_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the command is completed."""
    command = env.command_manager.get_term(command_name)
    return command.timestep_counter >= command.retargeted_horizon - 1
