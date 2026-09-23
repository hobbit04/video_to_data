"""Check the derived constants against the values measured by hand (no Isaac import)."""
import numpy as np, glob, re, torch, pyarrow.parquet as pq
R = '/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/'
def const(path, name):
    return float(re.search(rf'^{name} = ([\d.]+)', open(R+path).read(), re.M).group(1))
VAR_FLOOR, VAR_CEIL = const('rewards.py','_DERIVED_VAR_FLOOR'), const('rewards.py','_DERIVED_VAR_CEIL')
FRAC = const('terminations.py','_DERIVED_THRESHOLD_FRACTION')
P_FLOOR, O_FLOOR = const('terminations.py','_DERIVED_POSITION_FLOOR'), const('terminations.py','_DERIVED_ORIENTATION_FLOOR')

base='/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_refined/robot_name=sharpa_wave/'
t=pq.read_table(glob.glob(base+'*.parquet')[0]); col=lambda n: np.array(t.column(n)[0].as_py())
pos=torch.tensor(col('object_body_position')).float(); quat=torch.tensor(col('object_body_wxyz')).float()
travel = torch.cdist(pos.transpose(0,1), pos.transpose(0,1)).amax().item()
dots = torch.einsum("ibk,jbk->bij", quat, quat).abs().clamp(max=1.0)
rotation = (2.0*torch.acos(dots)).amax().item()
print(f"reference_object_travel   = {travel:.4f} m")
print(f"reference_object_rotation = {rotation:.4f} rad\n")
var = min(max(travel**2, VAR_FLOOR), VAR_CEIL)
pt  = max(FRAC*travel, P_FLOOR); ot = max(FRAC*rotation, O_FLOOR)
print(f"A-1  var                  0.10  ->  {var:.4f}")
print(f"A-2  position_threshold   0.20  ->  {pt:.4f} m")
print(f"A-2  orientation_threshold 0.70 ->  {ot:.4f} rad\n")

p0=pos[:,0,:].numpy(); q0=quat[:,0,:].numpy()
def Rm(w):
    w0,x,y,z=w
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w0),2*(x*z+y*w0)],[2*(x*y+z*w0),1-2*(x*x+z*z),2*(y*z-x*w0)],[2*(x*z-y*w0),2*(y*z+x*w0),1-2*(x*x+y*y)]])
Rs=np.stack([Rm(q) for q in q0]); U=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]],float)
K=p0[:,None,:]+np.einsum('tij,kj->tki',Rs,U); Kf=np.repeat((p0[0][None,:]+(Rs[0]@U.T).T)[None],len(p0),0)
d2=((K-Kf)**2).sum(-1)
print("무동작 정책 (상자를 frame 0에 고정):")
for name,v in (("var=0.1 (기존)",0.1),("derived",var)):
    h=np.exp(-d2/v).mean(); print(f"  {name:16s} 보상 {h:.4f}   과제 보상 폭 {100*(1-h):5.1f}%")
dp=np.linalg.norm(p0-p0[0],axis=1)
fired = np.where(dp>pt)[0]
print(f"\n  최대 위치 오차 {dp.max():.4f} m vs 파생 임계 {pt:.4f} m")
print(f"  -> {'frame %d 에서 종료 (%d/%d 프레임 초과)'%(fired[0],len(fired),len(dp)) if len(fired) else '종료되지 않음'}")
