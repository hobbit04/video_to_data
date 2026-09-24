set -u
cd /workspace/video_to_data/robotic_grounding
D=logs/rsl_rl/sharpa_v2d/2026-09-21_10-46-07_tissue_box_refined_8h
for IT in 200 600 1200 2000 2800 3400 3800 4078; do
  echo "########## checkpoint ${IT} ##########"
  python scripts/rsl_rl/diag_policy.py --headless --num_envs 32 --steps 518 \
    --motion_file ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave \
    --ckpt /workspace/video_to_data/robotic_grounding/$D/model_${IT}.pt 2>&1 \
    | grep -a "lift ratio (actual\|env metric object_lift_achieved\|RIGHT wrench-support"
done
