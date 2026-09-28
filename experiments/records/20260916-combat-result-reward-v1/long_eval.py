"""Paired initial/step8 evaluation; reuse the existing no-intervention evaluator."""
from feedback_eval import install as install_feedback
from learning_eval import SCENE_POST_GAME_SEEDS


def install_initial():
    install_feedback(expected_step=0, game_seeds=SCENE_POST_GAME_SEEDS,
                     expected_temperature=0.8)


def install_step8():
    install_feedback(expected_step=8, game_seeds=SCENE_POST_GAME_SEEDS,
                     expected_temperature=0.8)
