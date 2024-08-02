from typing import Literal


def infer_metric_direction(metric_name: str) -> Literal["min", "max"]:
    if metric_name.startswith("loss"):
        return "min"
    elif "acc" in metric_name:
        return "max"
    elif "auroc" in metric_name:
        return "max"
    elif "mrr" in metric_name:
        return "max"
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")
