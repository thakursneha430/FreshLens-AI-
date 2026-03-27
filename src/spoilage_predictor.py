import numpy as np

def predict_shelf_life(freshness_prob):
    """
    freshness_prob = probability of fresh class (0–1)
    Returns estimated days before spoilage
    """

    # simple heuristic model (can upgrade later)
    max_days = 7

    days_left = int(freshness_prob * max_days)

    if days_left >= 5:
        tip = "Store in fridge to keep fresh longer."
    elif days_left >= 2:
        tip = "Consume soon."
    else:
        tip = "Use immediately!"

    return days_left, tip