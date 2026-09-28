#!/usr/bin/env bash
# A/B: combat scale. B/C: temperature. C/D: gamma. No signal filtering.
set -euo pipefail
record=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(git -C "${record}" rev-parse --show-toplevel)
temperature=0.4
gamma=0.95
case "${1:?A, B, C or D}" in
    A) scale=0 ;;
    B) scale=0.001 ;;
    C) scale=0.001; temperature=0.8 ;;
    D) scale=0.001; temperature=0.8; gamma=1.0 ;;
    *) exit 2 ;;
esac
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    if [[ "$1" == C ]]; then
        test "${SC2_COMBAT_TEMPERATURE_AUTHORIZED:-0}" = 1
    fi
    if [[ "$1" == D ]]; then
        test "${SC2_COMBAT_GAMMA_AUTHORIZED:-0}" = 1
    fi
    test "${SC2_COMBAT_REWARD_AUTHORIZED:-0}" = 1
    test "$(git -C "${source_dir}" rev-parse HEAD)" = "${SOURCE_SHA:?frozen SHA}"
    test -z "$(git -C "${source_dir}" status --porcelain)"
    (cd "${source_dir}" && sha256sum -c configs/system-v1.sha256)
fi
: "${RUN_ID:?unique run id}"
: "${OUTPUT_DIR:?unused output directory}"
: "${FEEDBACK_STORAGE_ROOT:?prepared asset mirror}"
: "${FEEDBACK_PYTHON_OVERLAY:?verified dependency path}"
: "${FEEDBACK_INCLUDE_ROOT:?verified Python headers}"
export FEEDBACK_CONTINUE_A4=1 FEEDBACK_A4_COVERAGE=1 FEEDBACK_A4_SCENE1=0
export FEEDBACK_RL_GPUS=${FEEDBACK_RL_GPUS:-4,5,6,7} SC2_FEEDBACK_RL_AUTHORIZED=${SC2_COMBAT_REWARD_AUTHORIZED:-0}
unset FEEDBACK_A4_SIGNAL_ARM FEEDBACK_RL_TEMPERATURE FEEDBACK_TEMPERATURE_AB
unset FEEDBACK_SECOND_PLAN FEEDBACK_SECOND_PLAN_VARIANT FEEDBACK_LEARNING_MODEL SC2_COMMAND_TRACE_DIR
exec bash "${source_dir}/experiments/records/20260911-execution-failure-feedback-v1/run_feedback_train4_node43.sh" A \
    "+ray_init.runtime_env.env_vars.SC2_COMBAT_REWARD_SCALE='${scale}'" \
    trainer.sc2_skip_zero_signal=False \
    trainer.sc2_signal_max_attempts=3 trainer.sc2_signal_max_batches=4 \
    actor_rollout_ref.rollout.temperature="${temperature}" \
    algorithm.gamma="${gamma}"
