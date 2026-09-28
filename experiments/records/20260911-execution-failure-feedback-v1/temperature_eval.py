"""A4, one previously failed L6 seed; online answers, temperature-only A/B."""
import os

import feedback_eval

GAME_SEEDS = [42613] * 8
SCENE_B8_SEEDS = [1436145665] * 8


def install():
    temperature = os.environ['SC2_FEEDBACK_TEMPERATURE_AB']
    if (temperature not in ('0.4', '0.8')
            or os.environ['SC2_FEEDBACK_LEARNING_MODEL'] != 'A4'
            or os.environ['SC2_EXECUTION_FAILURE_FEEDBACK'] != '1'):
        raise ValueError('Temperature comparison requires A4, feedback on, T0.4 or T0.8')
    feedback_eval.install(expected_step=4, game_seeds=GAME_SEEDS,
                          expected_temperature=float(temperature))


def install_scene_b8():
    """Current SceneB8 recheck, not the historical A4/feedback-on experiment."""
    temperature = os.environ['SC2_SCENE_TEMPERATURE']
    if (temperature not in ('0.4', '0.8')
            or os.environ['SC2_FEEDBACK_LEARNING_MODEL'] != 'SceneB8'
            or os.environ['SC2_EXECUTION_FAILURE_FEEDBACK'] != '0'):
        raise ValueError('Scene temperature comparison requires SceneB8, feedback off, T0.4 or T0.8')
    feedback_eval.install(expected_step=8, game_seeds=SCENE_B8_SEEDS,
                          expected_temperature=float(temperature))
