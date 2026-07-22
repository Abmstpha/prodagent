"""LangSmith tracing — active when LANGSMITH_TRACING=true, a no-op otherwise."""
import os

if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    try:
        from langsmith import traceable
    except ImportError:
        traceable = None
else:
    traceable = None

if traceable is None:
    def traceable(*d_args, **d_kwargs):  # noqa: F811
        if d_args and callable(d_args[0]):
            return d_args[0]
        return lambda f: f
