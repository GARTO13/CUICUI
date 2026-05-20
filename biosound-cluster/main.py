"""Compatibility entrypoint for hosting biosound-cluster with uvicorn.

This lets deployment guides run:

    uvicorn main:app --host 0.0.0.0 --port 8000

The actual FastAPI application lives in ``biosound_cluster.api``.
"""

from biosound_cluster.api import app


__all__ = ["app"]
