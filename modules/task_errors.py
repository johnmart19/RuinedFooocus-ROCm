"""Actionable descriptions for recoverable task failures."""


def describe_error(error):
    detail = f"{type(error).__name__}: {error}"
    lowered = detail.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return f"The operation timed out. Check the connection and retry. Incomplete Hub downloads are kept for resuming. ({detail})"
    if "out of memory" in lowered:
        return f"There was not enough memory. Reduce the model/context size or generation settings, then retry. ({detail})"
    return f"The operation failed. You can retry or choose another model. ({detail})"
