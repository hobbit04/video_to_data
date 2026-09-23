#!/bin/bash
# 8-hour PPO training for the gsplat-refined tissue_box_refined sequence.
#
#   tmux new -s v2d
#   bash out/train_tissue_box_refined_8h.sh
#   (detach: Ctrl-b d   reattach: tmux attach -t v2d)
#
# WHY 4079 ITERATIONS
#   Measured on this host: 4096 envs, no --video, steady state 7.019 s/iter
#   (n=29, 6.86-7.16 s), Isaac boot+shutdown 153 s, warm-up iteration 11.9 s.
#   (8*3600 - 153 - 12) / 7.019 = 4079.
#
# WHY THE CURRICULUM IS COMPRESSED
#   The shipped schedule spans 2000..15500 iterations and only drops virtual
#   object control (VOC) to 0.0 at iteration 14000 -- roughly 27 h of wall clock
#   here. Left at the default, an 8 h run ends at iteration 4079 with VOC still
#   at 0.75, i.e. the virtual controller is still carrying the box and the policy
#   has never once held it alone. The schedule below is the shipped one scaled by
#   4079/15500 = 0.2632, so VOC reaches 0.0 at iteration 3684 (~7.2 h) and the
#   last ~395 iterations (~47 min) run fully self-reliant.
#
#   This buys coverage of the whole curriculum at the cost of ~1/4 the samples
#   per stage. It is the only way to have a *finished* policy inside 8 h; a
#   longer run with the stock schedule would be better if you can afford it.
#
# WHY THESE REWARD OVERRIDES (the force-closure recipe, README "Ego-video")
#   Monocular reconstruction gives noisy contact positions/normals, so
#   contact_wrench_support_reward trains against noise. force_closure gates on
#   the reference "contact expected" label instead. Each contact term is zeroed
#   in BOTH places: env.rewards.<name>.weight is what is in force from step 0,
#   and the curriculum's rewards_<name> is what gets written on schedule.
#   Zeroing only one leaves the term live for part of training.
#
# NOT the paper's recipe: CHORD uses FlashSAC (2048 envs, ~2 h on an L40S).
# FlashSAC is not in this repo, so this is PPO via rsl_rl. Expect different
# sample efficiency and a different wall-clock/quality trade-off.
set -euo pipefail

CONTAINER=${CONTAINER:-rg-retarget}
SEQ=${SEQ:-tissue_box_refined}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERS=${MAX_ITERS:-4079}
RUN_NAME=${RUN_NAME:-${SEQ}_8h}
SCHEDULE='[526,921,1316,1711,2105,2500,2895,3290,3684,4079]'

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
echo "envs          : ${NUM_ENVS}"
echo "iterations    : ${MAX_ITERS}  (~8 h at 7.019 s/iter)"
echo "run name      : ${RUN_NAME}"
echo "curriculum    : ${SCHEDULE}"
echo
echo "CHECK THE FIRST 60 SECONDS OF OUTPUT:"
echo "  - 'Active Reward Terms' must show force_closure 5.0 and the three"
echo "    contact terms 0.0. A silent overwrite stays invisible until the"
echo "    policy simply never grasps."
echo "  - 'Iteration time' should settle near 7.0 s. If it is much higher,"
echo "    something else is on GPU 6 and the 8 h estimate no longer holds."
echo "  - '[v2d] reference object travel 0.1559 m, rotation 0.0963 rad' must"
echo "    appear at startup. var=null derives the keypoint reward variance"
echo "    from it. Without that line the motion did not load as expected."
echo "  - '[v2d] object_keypoints_tracking_exp: derived var=0.02432' appears"
echo "    LATER, at the first curriculum step that gives the term a non-zero"
echo "    weight (iteration 526 here) -- Isaac Lab does not call a reward term"
echo "    whose weight is 0, so the value cannot be derived before then."
echo "    If it never appears, the override did not take and the task is worth"
echo "    6.7% of the reward instead of 21.5% -- see FINDINGS.md A-1."
echo
echo "NOT enabled: the derived object_away_from_trajectory thresholds (A-2)."
echo "The derived 0.078 m budget sits inside this environment's own noise"
echo "floor -- under zero actions 18.9% of env-steps already exceed it, and"
echo "62% of episodes terminate. Fix the reference penetration (B-1) first."
echo

# --- Preflight: the paper's quality gate ------------------------------------
# The paper filters sequences on hand-object penetration BEFORE training, and
# the 8 h run that motivated all of this was spent on a sequence scoring
# 2.19 cm against a 2 cm gate.  data_assessor exits non-zero under --reject, so
# this costs about two minutes and refuses to start on data that would not have
# been in the training set.  stride=1 because the default stride=3 samples every
# third frame and can step over the deepest one.
#   SKIP_PREFLIGHT=1 to run anyway (say why in the run notes).
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  echo "Preflight: penetration + replay gate on ${SEQ} ..."
  if ! docker exec "${CONTAINER}" bash -lc "
    cd /workspace/video_to_data/robotic_grounding
    python scripts/data_assessor.py \
      --input_dir source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed \
      --sequence_pattern '^${SEQ}$' \
      --checks hand_penetration \
      --set hand_penetration.stride=1 \
      --reject --output_reject /tmp/rejected_${SEQ}.txt
  "; then
    echo
    echo "ABORT: ${SEQ} does not pass the quality gate."
    echo "  A failing score means the retargeted reference puts the hand inside"
    echo "  the object, so the trajectory the policy is asked to imitate is not"
    echo "  physically reachable. Fix the reference first -- see"
    echo "  scripts/analysis/FINDINGS.md B-1 -- or re-run with SKIP_PREFLIGHT=1"
    echo "  if you are deliberately training on known-bad data."
    exit 1
  fi
  echo "Preflight passed."
  echo
fi

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
  env.curriculum.fixed_timestep_curriculum.params.rewards_contact_wrench_support_reward=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_unintended_contact_penalty=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_missed_contact_penalty=0.0
"

echo
echo "Checkpoints (save_interval=200) are under"
echo "  robotic_grounding/logs/rsl_rl/sharpa_v2d/<timestamp>_${RUN_NAME}/"
echo
echo "Render the trained policy afterwards -- training runs without --video on"
echo "purpose, because the renderer costs roughly 2x per iteration:"
echo "  docker exec ${CONTAINER} bash -lc 'cd /workspace/video_to_data/robotic_grounding && \\"
echo "    python scripts/rsl_rl/eval.py --headless --task Sharpa-V2D-v0-Play \\"
echo "      --motion_file ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave \\"
echo "      --checkpoint logs/rsl_rl/sharpa_v2d/<run>/model_4000.pt --video'"
