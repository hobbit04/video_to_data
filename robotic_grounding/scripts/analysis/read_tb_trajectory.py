import sys, numpy as np, bisect
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea=EventAccumulator(sys.argv[1],size_guidance={'scalars':0}); ea.Reload()
S={t:{s.step:s.value for s in ea.Scalars(t)} for t in ea.Tags()['scalars']}
M='Metrics/dual_hands_object_tracking_command/'
st=np.array(sorted(S[M+'object_lift_ratio']))
g=lambda t: np.array([S[t][k] for k in sorted(S[t])]) if t in S else np.full(len(st),np.nan)
ep=g('Train/mean_episode_length'); k=26.0/np.maximum(ep,1)/0.05
lift=g(M+'object_lift_ratio'); ach=g(M+'object_lift_achieved'); ref=g(M+'object_lift_reference')
voc=g(M+'virtual_object_controller_scale_factor'); oa=g('Episode_Termination/object_away_from_trajectory')
to=g('Episode_Termination/time_out'); rew=g('Train/mean_reward'); obe=g(M+'object_body_position_error')
wr=g('Episode_Reward/contact_wrench_support_reward')*k/10.0
un=g('Episode_Reward/unintended_contact_penalty')*k/10.0
ms=g('Episode_Reward/missed_contact_penalty')*k/1.0
okp=g('Episode_Reward/object_keypoints_tracking_exp')
sched=[120,230,330,420,500,570,640,850,1100,1500]; W=[0.0,0.1,0.25,0.25,0.5,0.5,1.0,1.0,5.0,20.0]
print(f"{'iter':>5} {'VOC':>5} {'LIFT':>7} {'ach':>7} {'ref':>7} {'objAway':>8} {'timeout':>8} {'wrench':>7} {'unint':>6} {'miss':>6} {'objkp':>6} {'objErr':>8} {'reward':>8}")
print('-'*104)
for i in [0,100,200,300,400,500,600,640,700,800,850,900,1000,1100,1200,1300,1400,1499]:
    j=min(int(np.searchsorted(st,i)),len(st)-1)
    w=W[min(bisect.bisect_right(sched,st[j]),9)]
    o=okp[j]*k[j]/w if w>0 else float('nan')
    print(f"{st[j]:5d} {voc[j]:5.2f} {lift[j]:7.4f} {ach[j]:7.4f} {ref[j]:7.4f} {oa[j]:8.4f} {to[j]:8.4f} "
          f"{wr[j]:7.3f} {un[j]:6.3f} {ms[j]:6.3f} {o:6.3f} {obe[j]*100:6.2f}cm {rew[j]:8.1f}")
m=st>=640
print(f"\nVOC=0 이후(iter>=640):  LIFT 평균 {lift[m].mean():.4f} 최대 {lift[m].max():.4f}")
print(f"마지막 50 iteration:     LIFT {lift[-50:].mean():.4f}  achieved {ach[-50:].mean():.4f} m  ref {ref[-50:].mean():.4f} m")
print(f"wrench(정규화):          시작 {wr[:20].mean():.3f}  중간 {wr[len(wr)//2-10:len(wr)//2+10].mean():.3f}  끝 {wr[-50:].mean():.3f}  최대 {wr.max():.3f}")
