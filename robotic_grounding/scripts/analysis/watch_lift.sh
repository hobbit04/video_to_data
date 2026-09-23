#!/bin/bash
# Live one-line-per-iteration view of the numbers that matter during training.
#
#   bash out/watch_lift.sh out/train_lift_1500.log
#
# Run it in a second tmux pane while the training log is being written. Works on
# a finished log too, in which case it prints everything and exits (FOLLOW=0).
#
# WHY THIS EXISTS
#   rsl_rl prints a ~35-line block per iteration, so the three lift metrics
#   scroll past several times a minute and the trend is impossible to see. This
#   collapses each block to one row.
#
#   LIFT is the point of the run: it was 0.06 on the failed 8 h policy. The rest
#   is context -- objAway and timeout are the termination split; wrench and fc
#   are contact_wrench_support_reward and force_closure with their weight and
#   episode length divided out, so they read as 0..1 and stay comparable across
#   curriculum steps. Whichever of the two a run enables, the other stays 0.
#   A rising reward with both flat means the curriculum raised a weight, not
#   that anything improved -- which is exactly how the 8 h run read as success.
#
#   WRENCH_W / FC_W override the weights used to normalise those two columns if
#   a run departs from the 10.0 / 5.0 the scripts use.
set -euo pipefail

LOG=${1:-out/train_lift_1500.log}
FOLLOW=${FOLLOW:-1}
WRENCH_W=${WRENCH_W:-10.0}
FC_W=${FC_W:-5.0}

if [ ! -f "$LOG" ]; then
    echo "No log at $LOG."
    echo "Start training with the output teed to one:"
    echo "  bash out/train_tissue_box_lift_1500.sh 2>&1 | tee $LOG"
    exit 1
fi

if [ "$FOLLOW" = "1" ]; then READER=(tail -n +1 -f "$LOG"); else READER=(cat "$LOG"); fi

"${READER[@]}" | stdbuf -oL tr -d '\r' | awk -v wrench_w="$WRENCH_W" -v fc_w="$FC_W" '
  /Learning iteration/ { if (it != "") emit(); split($0, a, /iteration[ \t]+/); split(a[2], b, "/"); it = b[1] }
  /Mean episode length/ { eplen = $NF }
  /Mean reward/ { rew = $NF }
  /object_lift_ratio/ { ratio = $NF }
  /object_lift_achieved/ { ach = $NF }
  /object_lift_reference/ { ref = $NF }
  /virtual_object_controller/ { voc = $NF }
  /Episode_Termination\/time_out/ { to = $NF }
  /object_away_from_trajectory:/ { oa = $NF }
  /Episode_Reward\/force_closure/ { fc = $NF }
  /Episode_Reward\/contact_wrench_support_reward/ { wr = $NF }
  /Episode_Reward\/unintended_contact_penalty/ { un = $NF }
  /Episode_Reward\/missed_contact_penalty/ { ms = $NF }
  END { if (it != "") emit() }

  function emit(  nfc, nwr, k) {
    if (++n == 1)
      printf("%6s %6s %8s %9s %10s %8s %8s %7s %7s %8s %8s\n",
             "iter", "VOC", "LIFT", "achieved", "reference", "objAway", "timeout",
             "wrench", "fc", "contact", "reward");
    # Undo Isaac Lab normalisation: Episode_Reward = sum(f * w * dt) / 26.0.
    k   = (eplen > 0) ? 26.0 / (eplen * 0.05) : 0;
    nwr = (wrench_w > 0) ? wr * k / wrench_w : 0;
    nfc = (fc_w > 0) ? fc * k / fc_w : 0;
    printf("%6s %6s %8s %9s %10s %8s %8s %7.3f %7.3f %8.2f %8s%s\n",
           it, voc, ratio, ach, ref, oa, to, nwr, nfc, wr + un + ms, rew,
           (ratio + 0 > 0.30) ? "   <-- moved" : "");
    fflush();
  }
'
