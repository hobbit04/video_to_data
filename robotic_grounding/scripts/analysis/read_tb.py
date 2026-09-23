import sys, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea=EventAccumulator(sys.argv[1], size_guidance={'scalars':0}); ea.Reload()
S={t:{s.step:s.value for s in ea.Scalars(t)} for t in ea.Tags()['scalars']}
M='Metrics/dual_hands_object_tracking_command/'
def col(t):
    v=S[t]; return np.array(sorted(v)), np.array([v[k] for k in sorted(v)])
steps,_=col(M+'object_lift_ratio')
get=lambda t: col(t)[1]
lift=get(M+'object_lift_ratio'); ach=get(M+'object_lift_achieved'); ref=get(M+'object_lift_reference')
voc=get(M+'virtual_object_controller_scale_factor'); oa=get('Episode_Termination/object_away_from_trajectory')
to=get('Episode_Termination/time_out'); fc=get('Episode_Reward/force_closure')
okp=get('Episode_Reward/object_keypoints_tracking_exp'); rew=get('Train/mean_reward')
ep=get('Train/mean_episode_length'); obe=get(M+'object_body_position_error')
rw=get(M+'right_hand_wrist_position_error'); lw=get(M+'left_hand_wrist_position_error')
import bisect
sched=[120,230,330,420,500,570,640,850,1100,1500]; W=[0.0,0.1,0.25,0.25,0.5,0.5,1.0,1.0,5.0,20.0]
print(f"{'iter':>6} {'VOC':>6} {'LIFT':>7} {'ach':>7} {'ref':>7} {'objAway':>8} {'timeout':>8} {'fc_n':>6} {'okp_n':>6} {'objErr':>7} {'wr R/L':>12} {'reward':>8}")
print('-'*105)
for i in [0,100,200,300,400,500,600,640,700,800,850,900,1000,1100,1200,1300,1400,1499]:
    j=int(np.searchsorted(steps,i)); j=min(j,len(steps)-1)
    w=W[min(bisect.bisect_right(sched,steps[j]),9)]
    n=lambda r,wt: r*26.0/(wt*max(ep[j],1)*0.05) if wt>0 else float('nan')
    print(f"{steps[j]:6d} {voc[j]:6.3f} {lift[j]:7.4f} {ach[j]:7.4f} {ref[j]:7.4f} {oa[j]:8.4f} {to[j]:8.4f} "
          f"{n(fc[j],5.0):6.3f} {n(okp[j],w):6.3f} {obe[j]*100:6.2f}cm {rw[j]*100:5.1f}/{lw[j]*100:5.1f} {rew[j]:8.1f}")
k=slice(-50,None)
print(f"\n마지막 50 iteration 평균:  LIFT {lift[k].mean():.4f}   achieved {ach[k].mean():.4f} m   reference {ref[k].mean():.4f} m")
print(f"LIFT 전체 최대 {lift.max():.4f} (iteration {steps[int(lift.argmax())]})")
print(f"VOC=0 이후(iter>=640) LIFT: 평균 {lift[steps>=640].mean():.4f}  최대 {lift[steps>=640].max():.4f}")
