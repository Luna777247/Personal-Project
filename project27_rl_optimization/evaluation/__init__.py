"""
Evaluation module for RL agents.
Provides evaluation utilities and visualization tools.
"""

from .metrics import (
    RLMetrics,
    evaluate_agent,
    compare_agents,
    plot_learning_curve,
    plot_metrics_comparison,
    plot_episodes_statistics,
)

__all__ = [
    "RLMetrics",
    "evaluate_agent",
    "compare_agents",
    "plot_learning_curve",
    "plot_metrics_comparison",
    "plot_episodes_statistics",
]
