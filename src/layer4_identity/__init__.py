"""
src/layer4_identity/__init__.py
─────────────────────────────────────────────────────────────────────────────
Layer 4: Identity — package initialiser.

Exports the public API: FaceIdentifier (the main orchestrator).

The export is LAZY (PEP 562). Importing it eagerly meant that touching ANY
module in this package — even just to read a path constant — pulled in the
whole computer-vision stack: cv2, torch, insightface, faiss. That made the
Layer 8 API server depend on those packages simply to learn where the
identity store lives, which is slow and wrong: a machine serving the
dashboard should not need the face models installed at all.

`from src.layer4_identity import FaceIdentifier` still works exactly as
before — the heavy imports just happen at that moment instead of at package
import time.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers and IDEs only; never executed at runtime
    from src.layer4_identity.identifier import FaceIdentifier

__all__ = ["FaceIdentifier"]


def __getattr__(name: str):
    if name == "FaceIdentifier":
        from src.layer4_identity.identifier import FaceIdentifier
        return FaceIdentifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
