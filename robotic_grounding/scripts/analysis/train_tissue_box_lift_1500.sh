#!/bin/bash
# ~3-hour PPO run testing whether the A-1 and B-1 fixes changed anything.
#
#   tmux new -s v2d
#   bash out/train_tissue_box_lift_1500.sh
#   (detach: Ctrl-b d   reattach: tmux attach -t v2d)
#
# WHAT THIS IS TESTING
#   The previous 8 h run raised the object 8.9 mm where the reference raises it
#   154 mm -- a lift ratio of 0.058 -- while every logged metric looked healthy.
#   Three things have changed since, and this run asks whether they move that
#   number at all. There is exactly one number to watch:
#
#       Metrics/dual_hands_object_tracking_command/object_lift_ratio
#
#   It was effectively 0.06 before. If it is still near that at the end, the
#   remaining causes are elsewhere and a longer run is not the answer.
#
# WHY ONLY 1500 ITERATIONS
#   In the 8 h run every weight-normalised reward plateaued by iteration ~900
#   and then flatlined for 3,100 more (force_closure peaked at 0.690 and drifted
#   down to 0.650; the hand-tracking terms started at 0.985 and only declined).
#   There is no evidence more iterations help until something else changes, so
#   this buys a read on the fixes rather than another long flat run.
#   At the measured 7.019 s/iter with 4096 envs: 1500 * 7.019 + 165 = 2 h 58 m.
#
# WHAT CHANGED SINCE THE 8 H RUN -- THREE THINGS AT ONCE
#   Attribution between them will need ablations; this run only asks whether the
#   combination does anything.
#
#   A-1  var=null derives the keypoint reward's variance from the reference's
#        own extent (travel^2 = 0.0243 here) instead of the fixed 0.1, which was
#        sized for mocap clips that travel ~0.32 m. The task's reward band goes
#        from 6.7% to 21.5%: a do-nothing policy scored 0.9326 out of 1.0 before
#        and scores 0.7851 now. In return terms, completing the task is worth
#        +70 instead of +22, against the ~147 forfeited by terminating early --
#        so attempting the lift pays off up to a 48 percentage-point rise in
#        termination risk, against 15 before.
#
#   B-1  The sequence was re-retargeted with --surface_project, taking hand-object
#        penetration from 2.188 cm to 1.392 cm. It now passes the paper's 2 cm
#        gate (0/390 frames over, against 19/390) and the replay check. The
#        trajectory the policy is asked to imitate is physically reachable.
#
#   CURRICULUM  Back-loaded rather than proportionally compressed -- see below.
#
# WHY THE CURRICULUM IS BACK-LOADED, NOT COMPRESSED
#   The 8 h run scaled the shipped schedule by a constant, which left only 395
#   iterations after VOC reached 0 (the stock schedule leaves 6000). Scaling to
#   1500 iterations would leave 146, which is strictly worse. So the shape
#   changes instead of the scale:
#
#     - VOC reaches 0.0 at iteration 640, leaving 860 iterations (57% of the
#       budget) with the policy holding the object unaided, against 395 (9.7%).
#     - The object-tracking weight ramps 1.0 -> 5.0 -> 20.0 instead of jumping
#       20x in one step. In the 8 h run that single jump dropped the normalised
#       object reward from 0.835 to 0.418 within 16 iterations and it had only
#       recovered to 0.549 by the end -- a value-function shock the run never
#       had time to absorb.
#     - The two shocks are separated: VOC hits 0 at index 7, the weight starts
#       climbing at index 8. Previously both landed on the same iteration.
#
#   This is a deliberate departure from the paper's schedule shape. If you want
#   the compressed-shape comparison instead, set SCHEDULE/VOC/OBJW to the
#   commented block below.
#
# WHY THESE REWARD OVERRIDES (the force-closure recipe, README "Ego-video")
#   Unchanged from the 8 h run. Monocular reconstruction gives noisy contact
#   positions/normals, so contact_wrench_support_reward trains against noise.
#   force_closure gates on the reference "contact expected" label instead. Each
#   contact term is zeroed in BOTH places: env.rewards.<name>.weight is what is
#   in force from step 0, and the curriculum's rewards_<name> is what gets
#   written on schedule. Zeroing only one leaves the term live for part of
#   training.
#
# NOT the paper's recipe: CHORD uses FlashSAC (2048 envs, ~2 h on an L40S).
# FlashSAC is not in this repo, so this is PPO via rsl_rl.
set -euo pipefail

CONTAINER=${CONTAINER:-rg-retarget}
SEQ=${SEQ:-tissue_box_refined}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERS=${MAX_ITERS:-1500}
RUN_NAME=${RUN_NAME:-${SEQ}_lift_1500}

# index:        0    1    2    3    4    5     6     7     8     9
SCHEDULE=${SCHEDULE:-'[120,230,330,420,500,570,640,850,1100,1500]'}
VOC=${VOC:-'[1.0,0.75,0.5,0.25,0.1,0.05,0.01,0.0,0.0,0.0]'}
OBJW=${OBJW:-'[0.0,0.1,0.25,0.25,0.5,0.5,1.0,1.0,5.0,20.0]'}
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
echo "envs / iters  : ${NUM_ENVS} / ${MAX_ITERS}   (~$(( (MAX_ITERS * 7 + 165) / 3600 ))h $(( ((MAX_ITERS * 7 + 165) % 3600) / 60 ))m)"
echo "run name      : ${RUN_NAME}"
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
echo "  - 'Active Reward Terms' must show force_closure 5.0 and the three"
echo "    contact terms 0.0. A silent overwrite stays invisible until the"
echo "    policy simply never grasps."
echo "  - 'Iteration time' should settle near 7.0 s. Much higher means"
echo "    something else is on GPU 6 and the 3 h estimate no longer holds."
echo
echo "THEN WATCH, roughly every 100 iterations:"
echo "  object_lift_ratio       the whole point. 0.06 = unchanged from the"
echo "                          failed run. Above ~0.3 means something moved."
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
  env.rewards.force_closure.weight=5.0 \
  env.rewards.contact_wrench_support_reward.weight=0.0 \
  env.rewards.unintended_contact_penalty.weight=0.0 \
  env.rewards.missed_contact_penalty.weight=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.timestep_schedule=${SCHEDULE} \
  env.curriculum.fixed_timestep_curriculum.params.virtual_object_control_scale_factor=${VOC} \
  env.curriculum.fixed_timestep_curriculum.params.rewards_object_keypoints_tracking_exp=${OBJW} \
  env.curriculum.fixed_timestep_curriculum.params.rewards_contact_wrench_support_reward=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_unintended_contact_penalty=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_missed_contact_penalty=0.0
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
