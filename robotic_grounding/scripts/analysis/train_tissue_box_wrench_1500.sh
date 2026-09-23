#!/bin/bash
# ~3-hour PPO run with CHORD's actual contact-wrench guidance turned on.
#
#   tmux new -s v2d
#   bash out/train_tissue_box_wrench_1500.sh
#   (detach: Ctrl-b d   reattach: tmux attach -t v2d)
#
# WHAT THIS IS TESTING
#   Every run so far -- the 8 h one and the 1500-iteration one -- had the
#   paper's core reward switched OFF. CHORD is "Contact Wrench Guidance from
#   Human Demonstration"; contact_wrench_support_reward is that guidance, and it
#   was at weight 0.0, with force_closure 5.0 standing in. So none of those runs
#   were a reproduction of the method, only of the pipeline around it.
#
#   This turns it on, at the repo's own stock weights. One number decides it:
#
#       Metrics/dual_hands_object_tracking_command/object_lift_ratio
#
#   0.058 on the 8 h policy, 0.057 after A-1 + B-1. If it is still there at the
#   end, the contact wrench guidance is not what was missing either.
#
# WHY THE SUBSTITUTE WAS THERE, AND WHY IT IS BEING REMOVED NOW
#   README "Ego-video (monocular) sequences" zeroes this reward because
#   monocular reconstruction gives noisy contact positions and normals, so
#   matching live geometry against reference geometry trains against noise.
#   That was a reasonable default, but it was never measured on this clip, and
#   the reference has improved since (gsplat refinement, then --surface_project
#   de-penetration). Measured now, on the sequence this run uses:
#
#     hand    contact points within 5 mm of the object surface    median dist
#     right   91.6%                                               0.3 mm
#     left    84.8%                                               0.7 mm
#
#   Normals are all unit length, none degenerate, and the active contact set
#   flips only 0.1-0.2 links per frame, so it is temporally stable too. That is
#   not noise. (scripts/analysis/contact_quality.py)
#
# WHY THESE WEIGHTS
#   The repo's stock curriculum, i.e. what CHORD runs on mocap data:
#   contact_wrench 10.0, unintended_contact -10.0, missed_contact -1.0, and
#   force_closure off, since force_closure was only ever the stand-in. A
#   40-iteration smoke confirms the balance is sane rather than penalty
#   dominated -- wrench +1.70, unintended -0.48, missed -0.14, net +1.07, with
#   the wrench term at 0.315 of its ceiling and room to climb.
#
#   To keep force_closure on alongside it, for a both-on variant:
#     FORCE_CLOSURE=5.0 bash out/train_tissue_box_wrench_1500.sh
#
# WHY ONLY 1500 ITERATIONS
#   In the 8 h run every weight-normalised reward plateaued by iteration ~900
#   and then flatlined for 3,100 more. Measured 7.338 s/iter at 4096 envs on the
#   1500-iteration run, which took 3.05 h wall clock. The contact terms add
#   work, so check the iteration time in the first 90 seconds and lower
#   MAX_ITERS if it has drifted.
#
# WHAT IS CARRIED OVER FROM THE PREVIOUS RUN
#   A-1  var=null derives the keypoint reward's variance from the reference's
#        own extent (0.0243 here) instead of the fixed 0.1. Task reward band
#        6.7% -> 21.5%.
#   B-1  The sequence is the --surface_project one, 1.392 cm penetration,
#        passing the paper's 2 cm gate and the replay check.
#   CURRICULUM  Back-loaded: VOC reaches 0 at iteration 640, leaving 860 (57%)
#        unaided, and the object weight ramps 1 -> 5 -> 20 instead of jumping
#        20x in one step. Unchanged from the previous run so this is a
#        one-variable comparison against it.
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
RUN_NAME=${RUN_NAME:-${SEQ}_wrench_1500}
FORCE_CLOSURE=${FORCE_CLOSURE:-0.0}

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
echo "envs / iters  : ${NUM_ENVS} / ${MAX_ITERS}   (~$(( (MAX_ITERS * 74 / 10 + 165) / 3600 ))h $(( ((MAX_ITERS * 74 / 10 + 165) % 3600) / 60 ))m)"
echo "run name      : ${RUN_NAME}"
echo "contact terms : wrench 10.0 / unintended -10.0 / missed -1.0 / force_closure ${FORCE_CLOSURE}"
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
echo "    force_closure ${FORCE_CLOSURE}. This is the whole point of the run; if"
echo "    the wrench term reads 0.0 the override did not take and you are"
echo "    repeating the previous experiment."
echo "  - 'Iteration time' should settle near 7.0 s. Much higher means"
echo "    something else is on GPU 6 and the 3 h estimate no longer holds."
echo
echo "THEN WATCH, roughly every 100 iterations:"
echo "  object_lift_ratio       the whole point. 0.06 = unchanged from both"
echo "                          previous runs. Above ~0.3 means something moved."
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
