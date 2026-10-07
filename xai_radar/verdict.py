def compute_verdict(score: float, thresholds: dict[str, float] | None = None) -> str:
    if thresholds is None:
        thresholds = {"green": 0.7, "yellow": 0.4}
    if score >= thresholds["green"]:
        return "GREEN"
    elif score >= thresholds["yellow"]:
        return "YELLOW"
    else:
        return "RED"
