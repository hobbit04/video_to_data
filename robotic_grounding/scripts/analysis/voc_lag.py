"""How much of the observed VOC tracking error does the damping term explain?"""
import numpy as np, glob, pyarrow.parquet as pq
from scipy.interpolate import interp1d
base='/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave/'
t=pq.read_table(glob.glob(base+'*.parquet')[0])
p=np.array(t.column('object_body_position')[0].as_py())[:,0,:]
fps=float(t.column('fps')[0].as_py()); T=len(p)

# The env resamples to 518 frames and steps at dt = 0.05 s (motion_speed 0.5
# stretches a 13.0 s clip to 25.9 s).
src=np.arange(T)/fps; tgt=np.linspace(0, src[-1], 518)
pe=interp1d(src,p,axis=0)(tgt)
dt=0.05
v=np.linalg.norm(np.diff(pe,axis=0),axis=1)/dt
a=np.linalg.norm(np.diff(pe,axis=0,n=2),axis=1)/dt**2

k,d,m = 50.0,10.0,0.3
print(f"환경 시간축 기준 레퍼런스 물체 속도: 평균 {v.mean():.4f}  p95 {np.percentile(v,95):.4f}  최대 {v.max():.4f} m/s")
print(f"                          가속도: 평균 {a.mean():.4f}  최대 {a.max():.4f} m/s^2\n")
lag_damp = (d/k)*v
lag_acc  = (m/k)*a
print(f"감쇠항이 강제하는 정상상태 오차 (d/k)*v : 평균 {lag_damp.mean()*100:5.2f}  p95 {np.percentile(lag_damp,95)*100:5.2f}  최대 {lag_damp.max()*100:5.2f} cm")
print(f"가속에 필요한 오차        (m/k)*a : 평균 {lag_acc.mean()*100:5.2f}  최대 {lag_acc.max()*100:5.2f} cm")
tot=lag_damp[:len(lag_acc)]+lag_acc
print(f"합                                : 평균 {tot.mean()*100:5.2f}  최대 {tot.max()*100:5.2f} cm")
print(f"\n실측 (제로 액션, VOC=1.0)         : p50 3.13  p90 9.06  p99 10.95 cm")
print(f"k=500,d=30 예측 (d/k=0.06)        : 평균 {(0.06*v).mean()*100:5.2f}  최대 {(0.06*v).max()*100:5.2f} cm")
print(f"k=500,d=30 실측                   : p50 0.44  p90 1.13  p99 2.80 cm")
