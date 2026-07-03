"""
src/layer4_identity/__init__.py
─────────────────────────────────────────────────────────────────────────────
Layer 4: Identity — package initialiser.

Exports the public API: FaceIdentifier (the main orchestrator).
"""

from src.layer4_identity.identifier import FaceIdentifier

__all__ = ["FaceIdentifier"]
