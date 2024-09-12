from datetime import datetime
from typing import Any, List

import gin


@gin.configurable()
def get_time_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@gin.configurable()
def get_str(format: str, values: List[Any]) -> str:
    return format.format(*values)
