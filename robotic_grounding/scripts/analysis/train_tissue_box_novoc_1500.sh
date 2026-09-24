#!/bin/bash
# ~3-hour PPO run with the virtual object controller OFF from step one.
#
#   tmux new -s v2d
#   bash out/train_tissue_box_novoc_1500.sh
#   (detach: Ctrl-b d   reattach: tmux attach -t v2d)
#
# WHAT THIS IS TESTING
#   The action space is a residual on top of reference tracking, so a zero
#   action means "execute the demonstration exactly". Measured with the virtual
#   controller fully off, that alone lifts the box:
#
#     zero actions, all envs from frame 0        lift ratio 0.664
#     zero actions, random starts (as in train)  lift ratio 0.238
#     every policy trained so far                lift ratio 0.05 - 0.06
#
#   The demonstration works. Every policy we trained is roughly ten times worse
#   than doing nothing, so a solution already sits at the origin of the action
#   space and PPO walks away from it. Widening the reward band (A-1), fixing the
#   reference penetration (B-1) and turning on the paper's contact wrench
#   guidance all left the lift ratio untouched, because none of them addressed
#   that.
#
#   The hypothesis here: the VOC curriculum carries the object through exactly
#   the phase in which behaviour forms, so the cost of perturbing the grasp is
#   invisible until VOC decays -- by which point the policy has settled into
#   residuals tuned for contact rewards that do not hold the box. Together with
#   init_noise_std 0.1, which means the policy never starts at zero action, the
#   grasp is being shaken apart before anything penalises it.
#
#   So: VOC = 0 from iteration 0. The object is the policy's problem immediately.
#
#   WHAT SUCCESS LOOKS LIKE
#     object_lift_ratio near 0.24 under these training conditions -- that is what
#     zero actions score with random start frames. The three previous runs sat at
#     0.02 - 0.03 in their VOC=0 phase. Anything above ~0.15 means the
#     curriculum was the problem.
#
#   WHAT FAILURE LOOKS LIKE
#     It starts near 0.24 and collapses anyway. Then the curriculum is not what
#     breaks the grasp, and the next suspect is exploration: init_noise_std,
#     entropy_coef, and the residual scales in actions_cfg.
#
# WHAT IS HELD FIXED FROM THE PREVIOUS RUN
#   Everything except VOC, so this is a one-variable comparison against the
#   contact-wrench run: the same de-penetrated sequence (B-1), var=null (A-1),
#   the same contact weights, the same object-weight ramp, the same 1500
#   iterations. Only virtual_object_control_scale_factor changes, to all zeros,
#   with initial_virtual_object_control_curriculum_scale 0.0 so the very first
#   iteration is unaided too.
#
#   The 20-step post-reset VOC hold (virtual_object_control_decay_steps) is left
#   at its default, because the zero-action baselines above were measured with
#   it. It only matters for episodes that start mid-air, where it gives the
#   grasp a moment to settle before the object becomes live.
#
# EVERY CONTACT TERM IS SET IN BOTH PLACES
#   env.rewards.<name>.weight is what is in force from step 0, and the
#   curriculum's rewards_<name> is what gets written on schedule. Setting only
#   one leaves the term at the wrong weight for part of training -- which is how
#   a silently-disabled reward goes unnoticed for 8 hours.
#
# NOT the paper's recipe: CHORD uses FlashSAC (2048 envs, ~2 h on an L40S).
# FlashSAC is not in this repo, so this is PPO via rsl_rl.
set -euo pipefail

CONTAINER=${CONTAINER:-rg-retarget}
SEQ=${SEQ:-tissue_box_refined}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERS=${MAX_ITERS:-1500}
RUN_NAME=${RUN_NAME:-${SEQ}_novoc_1500}
FORCE_CLOSURE=${FORCE_CLOSURE:-0.0}

# index:        0    1    2    3    4    5     6     7     8     9
SCHEDULE=${SCHEDULE:-'[120,230,330,420,500,570,640,850,1100,1500]'}
VOC=${VOC:-'[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]'}
OBJW=${OBJW:-'[0.0,0.1,0.25,0.25,0.5,0.5,1.0,1.0,5.0,20.0]'}
# The schedule now only drives the object-weight ramp; VOC is flat at 0.
# Compressed-shape alternative (the 8 h run's shape, scaled to 1500):
#   SCHEDULE='[193,339,484,629,774,919,1064,1210,1355,1500]'
#   VOC='[1.0,0.75,0.5,0.25,0.1,0.05,0.025,0.01,0.0,0.0]'
#   OBJW='[0.0,0.1,0.25,0.25,0.5,0.5,1.0,1.0,1.0,20.0]'

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "Container ${CONTAINER} is not running. Start it with:"
    echo
    echo "  docker run -d --gpus '\"device=6,7\"' --entrypoint /bin/sleep \\"
    echo "    -e ACCEPT_EULA=Y -e HEADLESS=1 \\"
    echo "    -v /rlwrld1/home/wongyun_yu/repos/video_to_data:/workspace/video_to_data \\"
    echo "    -v /rlwrld-unified-checkpoints/wongyun_yu/v2d/weights:/workspace/mano_weights \\"
    echo "    --name ${CONTAINER} robotic-grounding:latest infinity"
    exit 1
fi

echo "sequence      : ${SEQ}"
echo "envs / iters  : ${NUM_ENVS} / ${MAX_ITERS}   (~$(( (MAX_ITERS * 74 / 10 + 165) / 3600 ))h $(( ((MAX_ITERS * 74 / 10 + 165) % 3600) / 60 ))m)"
echo "run name      : ${RUN_NAME}"
echo "contact terms : wrench 10.0 / unintended -10.0 / missed -1.0 / force_closure ${FORCE_CLOSURE}"
echo "VOC           : OFF from iteration 0   (zero-action baseline: lift ratio 0.24)"
echo "curriculum    : ${SCHEDULE}"
echo "VOC           : ${VOC}"
echo "object weight : ${OBJW}"
echo

# --- Preflight: the paper's quality gate ------------------------------------
# The 8 h run was spent on a sequence scoring 2.19 cm against a 2 cm gate. This
# costs about two minutes and refuses to start on data that would not have been
# in the training set. stride=1 because the default stride=3 samples every third
# frame and can step over the deepest one.
#   SKIP_PREFLIGHT=1 to run anyway (say why in the run notes).
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  echo "Preflight: penetration gate on ${SEQ} ..."
  if ! docker exec "${CONTAINER}" bash -lc "
    cd /workspace/video_to_data/robotic_grounding
    python scripts/data_assessor.py \
      --input_dir source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed \
      --sequence_pattern '^${SEQ}\$' \
      --checks hand_penetration \
      --set hand_penetration.stride=1 \
      --reject --output_reject /tmp/rejected_${SEQ}.txt
  "; then
    echo
    echo "ABORT: ${SEQ} does not pass the quality gate."
    echo "  Re-retarget it with --surface_project --surface_margin 0.005;"
    echo "  see scripts/analysis/FINDINGS.md B-1."
    exit 1
  fi
  echo "Preflight passed."
  echo
fi

echo "CHECK THE FIRST 90 SECONDS OF OUTPUT:"
echo "  - '[v2d] reference object travel 0.1559 m, rotation 0.0963 rad'"
echo "    confirms the motion loaded and gives var=null something to derive from."
echo "  - 'Active Reward Terms' must show contact_wrench_support_reward 10.0,"
echo "    unintended_contact_penalty -10.0, missed_contact_penalty -1.0 and"
echo "    force_closure ${FORCE_CLOSURE}."
echo "  - VOC must read 0.000 in the very first iteration block. If it reads"
echo "    1.000 the override did not take and you are repeating the previous"
echo "    experiment."
echo "  - 'Iteration time' should settle near 7.0 s. Much higher means"
echo "    something else is on GPU 6 and the 3 h estimate no longer holds."
echo
echo "THEN WATCH, roughly every 100 iterations:"
echo "  object_lift_ratio       the whole point. Zero actions score 0.24 under"
echo "                          these conditions; the last three runs sat at"
echo "                          0.02-0.03. Watch whether it starts high and"
echo "                          collapses, or never gets there at all."
echo "  contact_wrench_support  should climb above the 0.315-of-ceiling the"
echo "                          smoke reached. Flat means the reward is not"
echo "                          learnable on this reference."
echo "  object_lift_achieved    in metres, against object_lift_reference."
echo "  '[v2d] object_keypoints_tracking_exp: derived var=0.02432' appears at"
echo "    iteration 120, the first curriculum step that gives the term a"
echo "    non-zero weight. Isaac Lab does not call a reward term whose weight"
echo "    is 0, so it cannot be derived before then. If it never appears, the"
echo "    var=null override did not take."
echo

docker exec "${CONTAINER}" bash -lc "
set -eo pipefail
cd /workspace/video_to_data/robotic_grounding
python scripts/rsl_rl/train.py \
  --headless --task Sharpa-V2D-v0 \
  --motion_file ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave \
  --num_envs ${NUM_ENVS} --max_iterations ${MAX_ITERS} \
  --logger tensorboard --run_name ${RUN_NAME} \
  env.rewards.object_keypoints_tracking_exp.params.var=null \
  env.rewards.contact_wrench_support_reward.weight=10.0 \
  env.rewards.unintended_contact_penalty.weight=-10.0 \
  env.rewards.missed_contact_penalty.weight=-1.0 \
  env.rewards.force_closure.weight=${FORCE_CLOSURE} \
  env.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.timestep_schedule=${SCHEDULE} \
  env.curriculum.fixed_timestep_curriculum.params.virtual_object_control_scale_factor=${VOC} \
  env.curriculum.fixed_timestep_curriculum.params.rewards_object_keypoints_tracking_exp=${OBJW} \
  env.curriculum.fixed_timestep_curriculum.params.rewards_contact_wrench_support_reward=10.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_unintended_contact_penalty=-10.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_missed_contact_penalty=-1.0
"

echo
echo "Checkpoints (save_interval=200) are under"
echo "  robotic_grounding/logs/rsl_rl/sharpa_v2d/<timestamp>_${RUN_NAME}/"
echo
echo "Render the trained policy afterwards -- training runs without --video on"
echo "purpose, because the renderer costs roughly 2x per iteration. Note eval.py"
echo "does not clamp viewer.env_index, so it needs the override below with"
echo "fewer than 7 envs:"
echo "  docker exec ${CONTAINER} bash -lc 'cd /workspace/video_to_data/robotic_grounding && \\"
echo "    python scripts/rsl_rl/eval.py --headless --task Sharpa-V2D-v0-Play \\"
echo "      --motion_file ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave \\"
echo "      --num_envs 1 --video --video_length 400 env.viewer.env_index=0 \\"
echo "      --checkpoint <path-to-model_*.pt>'"
echo
echo "Or read the lift directly, without rendering:"
echo "  docker exec ${CONTAINER} bash -lc 'cd /workspace/video_to_data/robotic_grounding && \\"
echo "    python scripts/rsl_rl/diag_policy.py --headless --num_envs 32 --steps 518 \\"
echo "      --motion_file ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave \\"
echo "      --ckpt <path-to-model_*.pt>'"
