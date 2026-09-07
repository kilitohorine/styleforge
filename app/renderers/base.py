from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Renderer(Protocol):
    modality: str

    def estimate_cost(self) -> float: ...
    def run(self, source: Path, style_id: str) -> tuple[np.ndarray, dict]: ...
