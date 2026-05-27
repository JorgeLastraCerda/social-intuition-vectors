from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
