"""自定义工具模块"""
from .game_tools import (
    generate_character,
    generate_clue,
    validate_clue_consistency,
    generate_plot_branch,
    calculate_suspicion_level,
    check_game_completion
)

__all__ = [
    "generate_character", "generate_clue", "validate_clue_consistency",
    "generate_plot_branch", "calculate_suspicion_level", "check_game_completion"
]
