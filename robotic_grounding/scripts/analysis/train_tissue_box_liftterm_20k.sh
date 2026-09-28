#!/bin/bash
# Stock 20,000-iteration PPO run plus ONE change: a lift-based failure termination.
#
#   tmux new -s v2d_lift
#   bash scripts/analysis/train_tissue_box_liftterm_20k.sh
#   (detach: Ctrl-b d   reattach: tmux attach -t v2d_lift)
#
# WHAT THIS IS TESTING
#   FINDINGS.md section 0: the demonstration lifts the box unaided (zero-action
#   lift ratio 0.66), and every trained policy is ~10x worse because the
#   objective makes lifting the risky choice -- object_away_from_trajectory
#   fires when an imperfect grasp displaces the box, and a termination costs
#   -100 plus the remaining reward stream, so "hold the box down" is safe.
#
#   Tightening object_away does not fix that (A-2): a flat 5 cm threshold cuts
#   70% of zero-action episodes short, because the hands push the box 3-10 cm
#   off the reference while carrying it. The deviation cannot separate a
#   non-lift from a noisy carry on a 15 cm clip.
#
#   So this run inverts the sign of the failure signal instead. The new
#   object_lift_failed termination fires when the reference, lagged by 40 env
#   steps (2 s), has risen at least 5 cm above the reset height and the
#   object's own rise is below 30% of that (running maxima, so an episode that
#   starts mid-air is judged on the lift remaining from there). The lag is
#   measured: the demonstration's own box starts rising 1-2 s after the
#   reference and then catches up. Under this objective, NOT lifting is what
#   gets terminated.
#
#   Note the VOC does NOT carry the box for this clip: at VOC=1.0 with zero
#   actions the lift ratio is 0.27 and a third of episodes fall below 0.3 x
#   reference, because the hands (60 N) overpower the controller (1.5 N at
#   3 cm). So the term is live from iteration 0, not only after VOC decays.
#
#   Everything else is the stock recipe as in train_tissue_box_stock_20k.sh:
#   stock curriculum (VOC 0 at 14000), stock PPO, contact wrench guidance on,
#   object_away kept at 0.2 m / 0.7 rad, var=null.
#
# WHAT SUCCESS LOOKS LIKE
#   object_lift_ratio holding well above the 0.05 the stock run sits at, in
#   the VOC = 0 phase (14000+). Episode_Termination/object_lift_failed should
#   start near zero (VOC carries the box), rise as VOC decays, and then fall
#   as the policy learns to lift rather than to avoid contact.
#
# WHAT FAILURE LOOKS LIKE
#   object_lift_failed stays high and the policy finds another way to dodge
#   it, or lift_ratio still decays -- then the objective is not the whole story.
#
# GPU
#   GPU=7 by default. The GPU is made exclusive to the process with
#   CUDA_VISIBLE_DEVICES inside the container (GPU 6 -> 0, GPU 7 -> 1), so the
#   RTX multi-GPU init does not touch the other card, and --device is cuda:0.
set -euo pipefail

CONTAINER=${CONTAINER:-rg-retarget}
SEQ=${SEQ:-tissue_box_refined}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERS=${MAX_ITERS:-20000}
GPU=${GPU:-7}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-${SEQ}_liftterm_20k}
REF_LIFT_MIN=${REF_LIFT_MIN:-0.05}
ACH_RATIO_MIN=${ACH_RATIO_MIN:-0.3}
LAG_STEPS=${LAG_STEPS:-40}
VAR=${VAR:-null}

case "${GPU}" in
  6) CVD=0 ;;
  7) CVD=1 ;;
  *) echo "GPU must be 6 or 7 (the user's allocation); got ${GPU}"; exit 1 ;;
esac

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
echo "envs / iters  : ${NUM_ENVS} / ${MAX_ITERS}   (~$(( MAX_ITERS * 68 / 10 / 3600 ))h at 6.8 s/iter)"
echo "GPU           : ${GPU}   (CUDA_VISIBLE_DEVICES=${CVD} in the container, --device cuda:0)"
echo "seed          : ${SEED}"
echo "run name      : ${RUN_NAME}"
echo "lift term     : ON  reference_lift_min ${REF_LIFT_MIN} m, achieved_lift_ratio_min ${ACH_RATIO_MIN}, lag_steps ${LAG_STEPS}"
echo "object_away   : STOCK 0.2 m / 0.7 rad (kept)"
echo "keypoint var  : ${VAR}"
echo "curriculum    : STOCK  VOC 1.0 -> 0.0 at 14000, obj weight 20 at 15500"
echo "contact terms : STOCK  wrench 10.0 / unintended -10.0 / missed -1.0 / force_closure 0.0"
echo

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
    echo "ABORT: ${SEQ} does not pass the quality gate (see FINDINGS.md B-1)."
    exit 1
  fi
  echo "Preflight passed."
  echo
fi

echo "CHECK THE FIRST 90 SECONDS OF OUTPUT:"
echo "  - 'Using device: cuda:0' and only GPU ${GPU} gaining memory on the host."
echo "  - 'Active Termination Terms' lists object_lift_failed alongside time_out,"
echo "    hand_wrist_away_from_trajectory and object_away_from_trajectory."
echo "  - 'Active Reward Terms' shows contact_wrench_support_reward 10.0, force_closure 0.0."
echo
echo "THEN WATCH:"
echo "  Episode_Termination/object_lift_failed   the new signal. ~0 while VOC=1,"
echo "                                           rising as VOC decays, then falling if it works."
echo "  object_lift_ratio                        against the stock run on GPU 6 (0.05)."
echo "  Episode_Termination/object_away_from_trajectory   should not explode."
echo

docker exec -e CUDA_VISIBLE_DEVICES="${CVD}" "${CONTAINER}" bash -lc "
set -eo pipefail
cd /workspace/video_to_data/robotic_grounding
python scripts/rsl_rl/train.py \
  --headless --device cuda:0 --task Sharpa-V2D-v0 \
  --motion_file ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave \
  --num_envs ${NUM_ENVS} --max_iterations ${MAX_ITERS} --seed ${SEED} \
  --logger tensorboard --run_name ${RUN_NAME} \
  env.terminations.object_lift_failed.params.enabled=true \
  env.terminations.object_lift_failed.params.reference_lift_min=${REF_LIFT_MIN} \
  env.terminations.object_lift_failed.params.achieved_lift_ratio_min=${ACH_RATIO_MIN} \
  env.terminations.object_lift_failed.params.lag_steps=${LAG_STEPS} \
  env.rewards.object_keypoints_tracking_exp.params.var=${VAR} \
  env.rewards.contact_wrench_support_reward.weight=10.0 \
  env.rewards.unintended_contact_penalty.weight=-10.0 \
  env.rewards.missed_contact_penalty.weight=-1.0 \
  env.rewards.force_closure.weight=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_contact_wrench_support_reward=10.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_unintended_contact_penalty=-10.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_missed_contact_penalty=-1.0
"

echo
echo "Checkpoints (save_interval=200) are under"
echo "  robotic_grounding/logs/rsl_rl/sharpa_v2d/<timestamp>_${RUN_NAME}/"
echo
echo "Read the lift without rendering (any checkpoint, while training continues):"
echo "  docker exec -e CUDA_VISIBLE_DEVICES=${CVD} ${CONTAINER} bash -lc 'cd /workspace/video_to_data/robotic_grounding && \\"
echo "    python scripts/rsl_rl/diag_policy.py --headless --device cuda:0 --num_envs 32 --steps 518 \\"
echo "      --motion_file ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave \\"
echo "      --ckpt <path-to-model_*.pt>'"
