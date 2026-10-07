import json
import os
from datetime import datetime, timezone
from typing import Any


def export_audit(
    result: dict[str, Any],
    path: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "consistency_score": result.get("consistency_score"),
        "verdict": result.get("verdict"),
        "metrics": result.get("metrics"),
        "metadata": metadata or {},
    }
    heatmaps = {}
    for key in ["gradcam", "shap", "integrated_gradients"]:
        arr = result.get(key)
        if arr is not None:
            heatmaps[key] = arr.tolist()
    payload["heatmaps"] = heatmaps

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
