#!/usr/bin/env bash
# Authorized controllers own the equal game budget and deadline.
set -euo pipefail
record_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record_dir}" rev-parse --show-toplevel)
case "${1:?A or B}" in
    A) skip=False ;;
    B) skip=True ;;
    *) exit 2 ;;
esac
authorization=${SC2_ZERO_SIGNAL_AUTHORIZED:-0}
unset FEEDBACK_A4_SIGNAL_ARM
if [[ "${SC2_SCENE_SAMPLING:-0}" == 1 ]]; then
    authorization=${SC2_SCENE_SAMPLING_AUTHORIZED:-0}
    skip=False
    export FEEDBACK_A4_SIGNAL_ARM=${1}
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    test "${authorization}" = 1
    test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?set frozen SHA}"
    test -z "$(git -C "${source_dir}" status --porcelain)"
    (cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
fi
: "${RUN_ID:?set unique RUN_ID}"
: "${OUTPUT_DIR:?set unused run directory}"
export FEEDBACK_CONTINUE_A4=1 FEEDBACK_A4_COVERAGE=1 FEEDBACK_A4_SCENE1=0
export FEEDBACK_RL_GPUS=0,1,2,3 SC2_FEEDBACK_RL_AUTHORIZED=${authorization}
unset FEEDBACK_RL_TEMPERATURE FEEDBACK_TEMPERATURE_AB
unset FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT FEEDBACK_LEARNING_MODEL SC2_COMMAND_TRACE_DIR
exec bash "${source_dir}/experiments/records/20260911-execution-failure-feedback-v1/run_feedback_train4_node43.sh" A \
    trainer.sc2_skip_zero_signal="${skip}" \
    trainer.sc2_signal_max_attempts=3 trainer.sc2_signal_max_batches=4
