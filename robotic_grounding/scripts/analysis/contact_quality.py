"""Is the reference contact geometry good enough to train contact_wrench_support_reward on?"""
import sys, glob, numpy as np, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0,'/workspace/video_to_data/robotic_grounding/scripts')
import filter_penetrations as F
import trimesh

SEQ = Path('/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave')
t = pq.read_table(glob.glob(str(SEQ/'*.parquet'))[0]); col = lambda n: np.array(t.column(n)[0].as_py())
hull, _ = F._load_hull(t.column('object_mesh_paths')[0].as_py()[0], {}, SEQ)
op, ow = col('object_body_position'), col('object_body_wxyz')
T = len(op)
print(f"frames {T}\n")
for side in ('right','left'):
    P  = col(f'mano_{side}_object_contact_positions')     # (T, L, 3) world
    N  = col(f'mano_{side}_object_contact_normals')
    ID = col(f'mano_{side}_object_contact_part_ids')
    act = (np.abs(P).sum(-1) > 1e-9)                       # zero = no contact sentinel
    n_per_frame = act.sum(1)
    print(f"--- {side} ---")
    print(f"  접촉 링크 수/프레임: 평균 {n_per_frame.mean():4.1f}  최대 {n_per_frame.max()}  0인 프레임 {(n_per_frame==0).sum()}/{T}")
    # distance of each active contact point to the object hull surface, in object frame
    d_all = []
    for i in range(0, T, 3):
        m = act[i]
        if not m.any(): continue
        pos = np.array(op[i][0], float); R = F._quat_wxyz_to_matrix(ow[i][0])
        loc = (P[i][m] - pos) @ R
        d_all.append(np.abs(trimesh.proximity.ProximityQuery(hull).signed_distance(loc)))
    d = np.concatenate(d_all)
    print(f"  물체 표면까지 거리:  중앙 {np.median(d)*1000:5.1f}  p90 {np.percentile(d,90)*1000:5.1f}  최대 {d.max()*1000:5.1f} mm")
    print(f"                       5mm 이내 {100*(d<0.005).mean():4.1f}%   1cm 이내 {100*(d<0.01).mean():4.1f}%")
    nn = np.linalg.norm(N[act], axis=-1)
    print(f"  normal 크기:         중앙 {np.median(nn):.3f}   0인 것 {100*(nn<1e-6).mean():4.1f}%")
    # temporal stability of the active set
    flips = (act[1:] != act[:-1]).sum(1)
    print(f"  접촉 on/off 전환:    프레임당 평균 {flips.mean():.2f} 링크")
