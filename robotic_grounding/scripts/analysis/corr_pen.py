"""Does the object's deviation track the reference's per-frame penetration?"""
import numpy as np, glob, sys, pyarrow.parquet as pq
from pathlib import Path
from scipy.interpolate import interp1d
sys.path.insert(0, '/workspace/video_to_data/robotic_grounding/scripts')
import filter_penetrations as F

seq = Path('/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave')
t = pq.read_table(glob.glob(str(seq/'*.parquet'))[0])
col = lambda n: t.column(n)[0].as_py()
rs, ls = F._get_shapes()
rh = F._HandShapeCache(rs, col('right_robot_frame_names')); lh = F._HandShapeCache(ls, col('left_robot_frame_names'))
hull, _ = F._load_hull(col('object_mesh_paths')[0], {}, seq)
rf, lf = col('robot_right_frames'), col('robot_left_frames')
op, ow = col('object_body_position'), col('object_body_wxyz')
pen = []
for i in range(len(rf)):
    pos = np.array(op[i][0], float); R = F._quat_wxyz_to_matrix(ow[i][0])
    caps = rh.world_spheres(rf[i]) + lh.world_spheres(lf[i])
    pen.append(F._max_hand_object_penetration(caps, hull, pos, R))
pen = np.array(pen)

dev = np.load('/workspace/video_to_data/robotic_grounding/out/diag_voc_pos_mean.npy')
# env timeline: 518 steps; the first 20 hold the clock, then tc advances 1 per step
tc = np.clip(np.arange(len(dev)) - 20, 0, None)
pen_env = interp1d(np.linspace(0, len(dev)-21, len(pen)), pen)(np.clip(tc, 0, len(dev)-21))

n = min(len(dev), len(pen_env)); dev, pen_env = dev[:n], pen_env[:n]
m = np.arange(n) > 40          # drop the reset hold + settle
print(f"프레임별 관통  평균 {pen.mean()*100:.2f}  최대 {pen.max()*100:.2f} cm")
print(f"편차(환경평균) 평균 {dev.mean()*100:.2f}  최대 {dev.max()*100:.2f} cm")
print(f"\n상관계수 (age>40): pearson r = {np.corrcoef(pen_env[m], dev[m])[0,1]:+.3f}")
from scipy.stats import spearmanr
print(f"                   spearman  = {spearmanr(pen_env[m], dev[m]).statistic:+.3f}")
lo, hi = pen_env[m] < np.median(pen_env[m]), pen_env[m] >= np.median(pen_env[m])
print(f"\n관통 하위 절반 구간 편차 평균 {dev[m][lo].mean()*100:5.2f} cm")
print(f"관통 상위 절반 구간 편차 평균 {dev[m][hi].mean()*100:5.2f} cm")
