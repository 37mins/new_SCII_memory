"""Evaluate actual model correction after two recorded answers; never PPO."""
from supply_order_eval import install as install_supply_evaluation


def install():
    install_supply_evaluation(prefix_answers=2)
