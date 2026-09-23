"""Penetration + IK task error for each surface_project margin."""
import sys, glob, numpy as np, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0,'/workspace/video_to_data/robotic_grounding/scripts')
import filter_penetrations as F

SEQ_DIR = Path('/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave')
rs, ls = F._get_shapes()
_hull_cache = {}

def measure(pq_path):
    t = pq.read_table(pq_path); col = lambda n: t.column(n)[0].as_py()
    hull, _ = F._load_hull(col('object_mesh_paths')[0], _hull_cache, SEQ_DIR)
    assert hull is not None, f"hull not loaded for {col('object_mesh_paths')[0]}"
    rh = F._HandShapeCache(rs, col('right_robot_frame_names'))
    lh = F._HandShapeCache(ls, col('left_robot_frame_names'))
    rf, lf = col('robot_right_frames'), col('robot_left_frames')
    op, ow = col('object_body_position'), col('object_body_wxyz')
    R=[];L=[];HH=[]
    for i in range(len(rf)):
        pos=np.array(op[i][0],float); Rm=F._quat_wxyz_to_matrix(ow[i][0])
        rc=rh.world_spheres(rf[i]); lc=lh.world_spheres(lf[i])
        R.append(F._max_hand_object_penetration(rc,hull,pos,Rm))
        L.append(F._max_hand_object_penetration(lc,hull,pos,Rm))
        HH.append(F._max_hand_hand_penetration(rc,lc))
    te_r=np.asarray(col('robot_right_frame_task_errors')); te_l=np.asarray(col('robot_left_frame_task_errors'))
    tips_r=np.asarray(col('mano_right_tips_distance')).reshape(len(rf),-1)
    tips_l=np.asarray(col('mano_left_tips_distance')).reshape(len(rf),-1)
    return dict(R=np.array(R),L=np.array(L),HH=np.array(HH),
                te=(te_r.mean(),te_l.mean()), te_max=(te_r.max(),te_l.max()),
                contact=((np.nanmin(tips_r,1)<0.01).mean()+(np.nanmin(tips_l,1)<0.01).mean())/2)

base='/workspace/video_to_data/robotic_grounding/out/pentest'
rows=[('none (baseline)','none')]+[(f'margin {m}',f'm{m}') for m in ('0.000','0.005','0.010','0.015','0.020')]
print(f"{'setting':<18} {'max':>6} {'mean':>6} {'p90':>6} {'>2cm':>9} {'>1cm':>9}  {'taskErr R/L':>13}")
print('-'*80)
for label,d in rows:
    p=glob.glob(f'{base}/{d}/sequence_id=*/robot_name=*/*.parquet')
    if not p: print(f"{label:<18}  (missing)"); continue
    m=measure(p[0]); worst=max(m['R'].max(),m['L'].max())
    over=((m['R']>0.02)|(m['L']>0.02)).sum()
    flag='  PASS' if worst<=0.02 else '  FAIL'
    both=np.maximum(m['R'],m['L'])
    print(f"{label:<18} {worst*100:5.2f} {both.mean()*100:6.2f} {np.percentile(both,90)*100:6.2f} "
          f"{over:5d}/{len(both)} {(both>0.01).sum():5d}/{len(both)}  "
          f"{m['te'][0]*100:5.2f}/{m['te'][1]*100:5.2f} cm{flag}")
