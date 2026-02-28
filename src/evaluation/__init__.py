"""Evaluation module – metrics, visualization, reporting."""

from .metrics import calculate_metrics, calculate_metrics_by_elevation, calculate_rmse
from .visualization import save_error_visualization, create_overall_visualization, plot_elevation_analysis
from .report import save_results_summary, save_worst_cases
