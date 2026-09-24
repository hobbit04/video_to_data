#!/bin/bash
# Full-length PPO run: the repo's stock 20,000-iteration recipe on the tissue box.
#
#   tmux new -s v2d20k
#   bash scripts/analysis/train_tissue_box_stock_20k.sh
#   (detach: Ctrl-b d   reattach: tmux attach -t v2d20k)
#
# WHAT THIS IS TESTING
#   Every run so far compressed the curriculum into 1,500-4,079 iterations. The
#   stock PPO config (config/sharpa_wave/agents/rsl_rl_ppo_cfg.py) says 20,000,
#   and the stock curriculum (v2d_hand_env_cfg.py) is written for that length:
#   VOC reaches 0 at iteration 14,000, leaving 6,000 unaided iterations, and the
#   object-keypoint weight jumps to 20 at 15,500. None of that has been run.
#
#   Two measurements say the short runs were under-trained in a specific way
#   (FINDINGS.md section F):
#     - the adaptive-KL schedule pins the learning rate at its 1e-5 floor for
#       ~45% of iterations (median 1.5e-5 against the nominal 1e-3), so each
#       iteration moves the policy very little;
#     - the action noise std is a learned parameter that drifts upward under
#       the entropy bonus, 0.10 -> 0.19 over the 8 h run, while the lift decays.
#
#   This script runs the stock recipe untouched except for the ego-video data
#   choices already established (contact wrench guidance on, de-penetrated
#   sequence, var=null). INIT_NOISE_STD lets a second copy test the exploration
#   hypothesis with a single knob; run both at once, one per GPU.
#
# HOW TO RUN TWO AT ONCE (GPU 6 = cuda:0, GPU 7 = cuda:1 inside the container)
#   DEVICE=cuda:0 RUN_NAME=tissue_box_stock_20k                bash scripts/analysis/train_tissue_box_stock_20k.sh
#   DEVICE=cuda:1 RUN_NAME=tissue_box_std003_20k INIT_NOISE_STD=0.03 bash scripts/analysis/train_tissue_box_stock_20k.sh
#
# TIME
#   Measured 6.3-7.4 s/iteration at 4096 envs on an A100, so 20,000 iterations
#   is roughly 36-41 h. Checkpoints land every 200 iterations (stock
#   save_interval), so the run can be read at any point with watch_lift.sh /
#   diag_policy.py without waiting for the end.
#
# NOT the paper's recipe: CHORD uses FlashSAC (2048 envs, ~2 h on an L40S) and
# perturbs the object with wrenches sampled from the human contact matrix.
# Neither is in this repo; this is rsl_rl PPO with no object perturbation.
set -euo pipefail

CONTAINER=${CONTAINER:-rg-retarget}
SEQ=${SEQ:-tissue_box_refined}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERS=${MAX_ITERS:-20000}
DEVICE=${DEVICE:-cuda:1}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-${SEQ}_stock_20k}
# Empty keeps the stock 0.1. The exploration-hypothesis variant uses 0.03: with
# zero actions the demonstration lifts the box (ratio 0.66), a 0.1 std costs 61%
# of that before the first gradient step (FINDINGS.md section 0).
INIT_NOISE_STD=${INIT_NOISE_STD:-}
# The keypoint reward variance. null derives it from the reference travel
# (0.0243 here; A-1). Set VAR=0.1 to run the exact stock value instead.
VAR=${VAR:-null}

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

EXTRA_AGENT=()
if [ -n "${INIT_NOISE_STD}" ]; then
    EXTRA_AGENT+=("agent.policy.init_noise_std=${INIT_NOISE_STD}")
fi

echo "sequence      : ${SEQ}"
echo "envs / iters  : ${NUM_ENVS} / ${MAX_ITERS}   (~$(( MAX_ITERS * 68 / 10 / 3600 ))h at 6.8 s/iter)"
echo "device        : ${DEVICE}   (cuda:0 = GPU 6, cuda:1 = GPU 7)"
echo "seed          : ${SEED}"
echo "run name      : ${RUN_NAME}"
echo "init std      : ${INIT_NOISE_STD:-0.1 (stock)}"
echo "keypoint var  : ${VAR}"
echo "curriculum    : STOCK  [2000,3500,...,15500], VOC 1.0 -> 0.0 at 14000, obj weight 20 at 15500"
echo "contact terms : STOCK  wrench 10.0 / unintended -10.0 / missed -1.0 / force_closure 0.0"
echo

# --- Preflight: the paper's quality gate ------------------------------------
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
echo "  - '[INFO][AppLauncher]: Using device: ${DEVICE}' -- the run is on the GPU you meant."
echo "  - '[v2d] reference object travel 0.1559 m, rotation 0.0963 rad'."
echo "  - 'Active Reward Terms' shows contact_wrench_support_reward 10.0, force_closure 0.0."
echo "  - 'Mean action noise std: ${INIT_NOISE_STD:-0.1}' in the first iteration block."
echo "  - VOC reads 1.000 (stock curriculum starts assisted)."
echo
echo "THEN WATCH (every ~500 iterations; bash scripts/analysis/watch_lift.sh <logdir>):"
echo "  object_lift_ratio        stock schedule: VOC is 0 from 14000, so the honest"
echo "                           read is iterations 14000-20000. Zero actions score"
echo "                           0.24 under random starts."
echo "  Mean action noise std    whether it drifts up from ${INIT_NOISE_STD:-0.1}. Every short run"
echo "                           grew it monotonically."
echo "  Loss/learning_rate       1e-5 means the KL schedule is at its floor again."
echo "  '[v2d] object_keypoints_tracking_exp: derived var=0.02432' appears at"
echo "    iteration 2000, the first step with a non-zero keypoint weight."
echo

docker exec "${CONTAINER}" bash -lc "
set -eo pipefail
cd /workspace/video_to_data/robotic_grounding
python scripts/rsl_rl/train.py \
  --headless --device ${DEVICE} --task Sharpa-V2D-v0 \
  --motion_file ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave \
  --num_envs ${NUM_ENVS} --max_iterations ${MAX_ITERS} --seed ${SEED} \
  --logger tensorboard --run_name ${RUN_NAME} \
  env.rewards.object_keypoints_tracking_exp.params.var=${VAR} \
  env.rewards.contact_wrench_support_reward.weight=10.0 \
  env.rewards.unintended_contact_penalty.weight=-10.0 \
  env.rewards.missed_contact_penalty.weight=-1.0 \
  env.rewards.force_closure.weight=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_contact_wrench_support_reward=10.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_unintended_contact_penalty=-10.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_missed_contact_penalty=-1.0 \
  ${EXTRA_AGENT[*]:-}
"

echo
echo "Checkpoints (save_interval=200) are under"
echo "  robotic_grounding/logs/rsl_rl/sharpa_v2d/<timestamp>_${RUN_NAME}/"
echo
echo "Read the lift without rendering (any checkpoint, while training continues):"
echo "  docker exec ${CONTAINER} bash -lc 'cd /workspace/video_to_data/robotic_grounding && \\"
echo "    python scripts/rsl_rl/diag_policy.py --headless --device ${DEVICE} --num_envs 32 --steps 518 \\"
echo "      --motion_file ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave \\"
echo "      --ckpt <path-to-model_*.pt>'"
