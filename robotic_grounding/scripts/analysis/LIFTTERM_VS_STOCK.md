# tissue-box 20k 학습: 기본 설정 run과 lift 종료 조건 run의 차이

두 run 모두 `Sharpa-V2D-v0`, 시퀀스 `tissue_box_refined`, PPO 20,000 iteration,
4096 envs, seed 42로 학습했다.

| run | 로그 디렉터리 | 학습 스크립트 |
|---|---|---|
| **기본 설정** | `logs/rsl_rl/sharpa_v2d/2026-09-24_09-15-27_tissue_box_stock_20k` | `train_tissue_box_stock_20k.sh` |
| **lift 종료 조건** | `logs/rsl_rl/sharpa_v2d/2026-09-24_21-19-42_tissue_box_liftterm_20k` | `train_tissue_box_liftterm_20k.sh` |

## 한 줄 요약

**두 run의 차이는 종료 조건 `object_lift_failed`를 켰는지 여부 하나뿐이다.**
각 run이 저장한 `params/env.yaml`과 `params/agent.yaml`을 diff하면 이 항목과
`log_dir`/`run_name` 외에는 한 줄도 다르지 않다. 코드 버전, seed, PPO 하이퍼파라미터,
reward 가중치, curriculum이 모두 같다.

## 1. 유일한 차이: `object_lift_failed` 종료 조건

### 무엇을 하는가

레퍼런스(사람 시연)는 물체를 들어 올렸는데 정책은 들어 올리지 못했을 때 에피소드를
종료한다. 구현은 `source/.../tasks/v2d/mdp/terminations.py`의 `ObjectLiftFailed`이고,
등록은 `v2d_hand_env_cfg.py`의 `TerminationsCfg.object_lift_failed`이다.

매 스텝 다음 조건을 검사한다.

```
lagged_ref  = (lag_steps 스텝 전 레퍼런스의 상승량)의 running max   # reset 시점 물체 높이 기준
achieved    = 실제 물체 상승량의 running max                     # 같은 기준

종료 ⇔ lagged_ref ≥ reference_lift_min  AND  achieved < achieved_lift_ratio_min × lagged_ref
```

| 파라미터 | 값 | 의미 |
|---|---|---|
| `reference_lift_min` | 0.05 m | 레퍼런스가 5 cm 이상 올라간 뒤부터만 검사 |
| `achieved_lift_ratio_min` | 0.3 | 실제 상승이 레퍼런스의 30% 미만이면 실패 |
| `lag_steps` | 40 스텝 (2 s) | 레퍼런스를 2초 늦춰 비교 |
| `enabled` | `true` (이 run만) | 기본값은 `false` |

- `time_out: false`인 종료 항이므로 기존 `termination_penalty`(가중치 −100)가 그대로
  붙는다. `termination_penalty`는 `termination_manager.terminated`, 즉 time-out이 아닌
  모든 종료를 반환한다(`mdp/rewards.py:437`). 따라서 들지 않은 정책은 −100을 받고,
  남은 에피소드의 보상도 잃는다.
- 두 높이 모두 reset 시점 물체 높이를 기준으로 재므로, 무작위 프레임에서 시작한
  에피소드에도 적용된다. 레퍼런스가 아직 올라가지 않은 구간에서는 항상 거짓이다.
- 40스텝 지연을 둔 이유는 측정에 있다. zero action(시연 그대로 재생)에서도 실제 상자는
  레퍼런스보다 1–2초 늦게 올라간다. 지연 없이 검사하면 시연 에피소드의 77%가
  종료되고, 40스텝 지연을 두면 frame 0 시작 기준 0.2%만 종료된다(FINDINGS.md F-5).

### 왜 넣었는가

FINDINGS.md 0절에서 확인한 메커니즘 때문이다. 기본 설정에서 실패 신호는
`object_away_from_trajectory`(물체가 레퍼런스에서 0.2 m / 0.7 rad 이상 벗어나면 종료)뿐이다.
불완전한 grasp로 0.3 kg 상자를 들면 물체가 흔들려 이 조건에 걸릴 가능성이 가장 높다.
그래서 PPO는 "물체를 건드리지 않는 것이 안전하다"를 학습했다. 이 임계값을 조이는 방법은
통하지 않는다. 5 cm로 줄이면 시연 자체의 70%가 잘린다(A-2).

`object_lift_failed`는 신호의 부호를 뒤집는다. 기본 설정에서는 **드는 것**이
종료 위험이었다면, 이 run에서는 **들지 않는 것**이 종료된다.

### 코드 변경 범위

- 기본값이 `enabled=False`여서 꺼져 있으면 항상 all-False를 반환한다. 따라서 이 항이
  추가된 코드로 기본 설정 run을 돌려도 동작은 원본과 같다. 기본 설정 run의
  `env.yaml`에는 이 항이 아예 기록되어 있지 않다.
- 켜는 방법은 명령줄 override 하나다.
  `env.terminations.object_lift_failed.params.enabled=true`

## 2. 두 run에 공통으로 적용된, NVIDIA 기본값과 다른 점

"기본 설정" run도 upstream(`1b22145f`)을 그대로 돌린 것은 아니다. 아래 항목은 두 run에
**똑같이** 들어가 있으므로 두 run 사이의 결과 차이를 설명하지는 않는다. 다만
"완전 기본값"이라고 부를 때는 알고 있어야 한다.

| 항목 | upstream 기본 | 두 run | 학습에 영향 |
|---|---|---|---|
| `object_keypoints_tracking_exp.var` | 0.1 | `null` → 레퍼런스 이동량에서 유도, **0.0243** | **있음** (FINDINGS A-1) |
| 레퍼런스 모션 | 원본 retarget | 손-물체 관통을 줄인 재retarget 버전 | **있음** (FINDINGS B-1, 데이터 변경) |
| `contact_wrench_support_reward` 10.0, `unintended_contact_penalty` −10.0, `missed_contact_penalty` −1.0, `force_closure` 0.0 (curriculum 포함) | 동일 | 명령줄에 명시 | 없음 (기본값과 같은 값을 명시했을 뿐) |
| `object_lift_*` 지표 | 없음 | 로깅만 추가 | 없음 (metric 전용, 보상에 쓰이지 않음) |
| `object_away_from_trajectory` | 0.2 m / 0.7 rad | 0.2 m / 0.7 rad | 없음 |
| curriculum, PPO(`init_noise_std` 0.1, adaptive KL 등) | — | 기본값 | 없음 |

upstream 0.1과 완전히 같게 돌리려면 `VAR=0.1`을 지정하면 된다.

## 3. 학습 로그상의 결과

**curriculum 시점 주의.** `FixedTimestepCurriculum`은 `bisect_right`로 인덱스를 고르므로,
값 목록의 k번째 항목은 `timestep_schedule[k-1]`부터 적용된다. 그래서 실제로는
**VOC가 12,500 it에 0이 되고, 물체 keypoint 가중치가 14,000 it에 1 → 20으로 바뀐다.**
15,500 경계에서는 아무것도 바뀌지 않는다. 학습 스크립트의 안내 문구("VOC 0 at 14000,
obj weight 20 at 15500")는 한 칸 늦게 적혀 있다. 이것은 upstream 동작이며 두 run에 동일하다.

TensorBoard 로그에서 읽은 값이다. **지표 정의부터 적는다.**

- `object_lift_ratio`: 에피소드 동안 실제 물체 상승량의 running max를 레퍼런스 상승량의
  running max로 나눈 값이다(`hand_object_commands.py:_update_lift_metrics`). 레퍼런스
  상승이 1 cm 미만인 에피소드는 0으로 기록된다. Isaac Lab은 command metric을 reset 시점에
  읽으므로, 한 iteration의 값은 그 iteration에 끝난 에피소드들의 종료 시점 값 평균이다.
- 학습 중 값이므로 **stochastic action(noise 포함)과 무작위 시작 프레임** 조건이다.
  FINDINGS 0절의 "frame 0, deterministic" 재생 수치와 직접 비교하는 값이 아니다.
- `Episode_Termination/*`: 가장 최근 에피소드가 해당 원인으로 끝난 env의 비율이다
  (FINDINGS C-2). `time_out`은 "조기 종료되지 않음"이라는 뜻이며 성공률이 아니다.

| 구간 | 지표 | 기본 설정 | lift 종료 조건 |
|---|---|---|---|
| VOC 감소 구간 (0–12,500 it) | `object_lift_ratio` 평균 | 0.074 | 0.566 |
| VOC = 0, 물체 가중치 1 (12,500–14,000 it) | `object_lift_ratio` 평균 | 0.020 | 0.771 |
| **VOC = 0, 물체 가중치 20 (14,000–20,000 it)** | `object_lift_ratio` 평균 | **0.028** | **0.636** |
| | `object_lift_achieved` 평균 | 0.0038 m | 0.0787 m |
| 마지막 500 it | `object_lift_ratio` 평균 | 0.048 | 0.556 |
| 19,999 it | 물체 위치 오차 | 1.73 cm | 1.34 cm |
| | `object_keypoints_tracking_exp` (episode 합) | 9.33 | 11.17 |
| | `object_away` 종료 비율 | 0.011 | 0.003 |
| | `object_lift_failed` 종료 비율 | — | 0.0004 |
| | mean reward | 319 | 373 |

해석:

1. **들기가 살아남았다.** 기본 설정 run은 VOC가 줄어들며 lift ratio가 0.25(2,000 it)에서
   0.03 이하로 떨어졌다. 이는 이전 run들과 같은 패턴이다. lift 종료 조건 run은 VOC가 0이
   된 뒤(12,500 it~)에도 0.6–0.8을 유지했다. 영상 확인 결과와도 일치한다.
2. **목표 함수 자체로도 더 좋다.** 기본 설정 run보다 물체 keypoint 보상, 물체 오차,
   `object_away` 종료율, mean reward가 모두 낫다. 기본 설정 run이 수렴한 "물체를 두는"
   해는 목표의 최적이 아니었다. `object_away`가 만든 국소 해였다(FINDINGS 0절의 결론).
3. **`object_lift_failed`는 이제 거의 발동하지 않는다.** 초기에는 0.12–0.14였고,
   물체 가중치가 20이 된 14,000 it 이후에는 0.0004–0.002로 내려갔다. 정책이 종료를 다른 방식으로 피한 것이
   아니라, 실제로 들어서 조건을 통과하고 있다는 뜻이다(achieved 0.07–0.10 m).

주의할 점:

- **lift ratio가 최고점 이후 줄어든다.** 13,000 it의 0.80에서 19,999 it의 0.57로
  내려갔다. 종료 조건은 30%만 요구하므로 그 이상을 유지할 압력이 약하다. 같은 기간에
  action noise std가 0.37에서 0.42로 커졌다(FINDINGS F-2). 학습 지표는 noise가 포함된
  값이라, 이 감소가 정책 평균의 퇴화인지 noise 증가 때문인지는 로그만으로 구분할 수 없다.
  최종 정책을 쓸 때는 12,000–14,000 it 체크포인트도 함께 비교하는 것이 좋다.
- 위 수치는 모두 학습 로그 값이다. 체크포인트를 frame 0부터 deterministic하게 재생한
  평가(env별 첫 에피소드만 집계)는 이 문서 작성 시점에 끝나지 않아 포함하지 않았다.

## 재현

```bash
# 기본 설정
bash scripts/analysis/train_tissue_box_stock_20k.sh          # GPU=6 기본

# lift 종료 조건 (위와 동일 + object_lift_failed 활성화)
bash scripts/analysis/train_tissue_box_liftterm_20k.sh       # GPU=7 기본
```

두 스크립트의 `train.py` 인자는 `--run_name`, GPU 지정 방식(`--device`), 아래 네 줄을
제외하면 같다. 기본 설정 스크립트의 선택 인자 `INIT_NOISE_STD`는 이 run에서 비워 두었다.
두 run의 `agent.yaml`이 동일하므로 `init_noise_std`는 둘 다 기본값 0.1이다.

```
env.terminations.object_lift_failed.params.enabled=true
env.terminations.object_lift_failed.params.reference_lift_min=0.05
env.terminations.object_lift_failed.params.achieved_lift_ratio_min=0.3
env.terminations.object_lift_failed.params.lag_steps=40
```
