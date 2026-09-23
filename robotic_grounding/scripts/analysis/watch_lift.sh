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
#   is context -- objAway and timeout are the termination split, fc is
#   force_closure with its weight and episode length divided out so it reads as
#   0..1 and is comparable across curriculum steps. A rising reward with a flat
#   fc means the curriculum raised a weight, not that anything improved.
set -euo pipefail

LOG=${1:-out/train_lift_1500.log}
FOLLOW=${FOLLOW:-1}

if [ ! -f "$LOG" ]; then
    echo "No log at $LOG."
    echo "Start training with the output teed to one:"
    echo "  bash out/train_tissue_box_lift_1500.sh 2>&1 | tee $LOG"
    exit 1
fi

if [ "$FOLLOW" = "1" ]; then READER=(tail -n +1 -f "$LOG"); else READER=(cat "$LOG"); fi

"${READER[@]}" | stdbuf -oL tr -d '\r' | awk '
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
  END { if (it != "") emit() }

  function emit(  nfc) {
    if (++n == 1)
      printf("%6s %6s %8s %9s %10s %8s %8s %7s %8s\n",
             "iter", "VOC", "LIFT", "achieved", "reference", "objAway", "timeout", "fc", "reward");
    nfc = (eplen > 0) ? fc * 26.0 / (5.0 * eplen * 0.05) : 0;
    printf("%6s %6s %8s %9s %10s %8s %8s %7.3f %8s%s\n",
           it, voc, ratio, ach, ref, oa, to, nfc, rew,
           (ratio + 0 > 0.30) ? "   <-- moved" : "");
    fflush();
  }
'
