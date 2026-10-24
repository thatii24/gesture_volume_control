import numpy as np

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x, y)."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def smooth_value(current_value, previous_value, alpha=0.2):
    """
    Applies exponential smoothing to a value.
    alpha: smoothing factor (0-1). Lower is smoother but slower.
    """
    return alpha * current_value + (1 - alpha) * previous_value
