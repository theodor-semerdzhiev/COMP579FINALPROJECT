from typing import List, Dict, Any
import numpy as np

def evaluate_knapsack_performance(trained_values: List[float],
                                  dp_values: List[float],
                                  greedy_values: List[float]) -> Dict[str, Any]:
    """
    Computes evaluation metrics comparing the RL model's solution values to the optimal (DP)
    and greedy baseline values. The returned dictionary contains both aggregated table metrics
    and detailed error statistics.

    Aggregated Table Metrics:
      - N: Number of problem instances.
      - Val: Sum of RL model’s solution values across all instances.
      - #opt: Count of instances where the RL solution exactly matches the DP optimum.
      - #highest: Count of instances where the RL solution is at least as high as both the DP and greedy solutions.
      - ValRatio: Ratio (in percent) of the RL total value to the DP total value.
    
    Detailed Error Metrics:
      - mean_absolute_error: Mean of the absolute differences between RL and DP values.
      - min_absolute_error: Minimum absolute error across instances.
      - max_absolute_error: Maximum absolute error across instances.
      - mean_percentage_error: Mean relative (percentage) error computed as |RL - DP| / DP for each instance (ignores instances with zero DP value).
      - mean_improvement_over_greedy: Mean relative improvement of RL over the greedy solution computed as (RL - greedy) / DP for each instance.

    Parameters:
      trained_values: List of solution values obtained from the RL model for each instance.
      dp_values: List of optimal (DP) solution values for each instance.
      greedy_values: List of solution values computed with a greedy heuristic for each instance.

    Returns:
      A dictionary with the following keys:
        - 'N': Number of instances.
        - 'Val': Total RL solution value.
        - '#opt': Count where RL exactly matches DP.
        - '#highest': Count where RL is at least as high as both DP and Greedy.
        - 'ValRatio': (Val_RL / total_DP_value) * 100 (%).
        - 'mean_absolute_error': Mean absolute error between RL and DP values.
        - 'min_absolute_error': Minimum absolute error.
        - 'max_absolute_error': Maximum absolute error.
        - 'mean_percentage_error': Mean percentage error relative to DP.
        - 'mean_improvement_over_greedy': Mean relative improvement of RL over Greedy.
    """
    if not (len(trained_values) == len(dp_values) == len(greedy_values)):
        raise ValueError("All input lists must have the same length.")
    
    n = len(trained_values)
    
    # Aggregated Metrics
    # Total value (sum) of RL and DP solutions
    total_rl_value = sum(trained_values)
    total_dp_value = sum(dp_values)
    
    # Count instances where RL solution equals the optimal DP solution
    count_opt = sum(1 for rl, dp in zip(trained_values, dp_values) if rl == dp)
    
    # Count instances where RL solution is at least as high as both DP and Greedy solutions
    # (Change >= to > if you require strictly higher values.)
    count_highest = sum(
        1 for rl, dp, gr in zip(trained_values, dp_values, greedy_values)
        if rl >= dp and rl >= gr
    )
    
    # Ratio of total RL value to total DP value, expressed as a percentage.
    val_ratio = (total_rl_value / total_dp_value * 100.0) if total_dp_value != 0 else 0.0
    
    # Detailed Error Metrics
    abs_errors = [abs(rl - dp) for rl, dp in zip(trained_values, dp_values)]
    # Avoid division by zero when computing percentage errors; assume error = 0 if dp == 0.
    perc_errors = [abs(rl - dp) / dp if dp != 0 else 0 for rl, dp in zip(trained_values, dp_values)]
    # Relative improvement of RL over greedy, based on DP as reference.
    improvement_over_greedy = [(rl - gr) / dp if dp != 0 else 0 for rl, gr, dp in zip(trained_values, greedy_values, dp_values)]
    
    performance_metrics = {
        # Aggregated metrics similar to your table
        'N': n,
        'Val': total_rl_value,
        '#opt': count_opt,
        '#highest': count_highest,
        'ValOptRatio': val_ratio,
        # Detailed error metrics
        'mean_absolute_error': np.mean(abs_errors),
        'min_absolute_error': np.min(abs_errors),
        'max_absolute_error': np.max(abs_errors),
        'mean_percentage_error': np.mean(perc_errors),
        'mean_improvement_over_greedy': np.mean(improvement_over_greedy)
    }
    
    return performance_metrics
