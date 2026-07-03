#!/usr/bin/env python3
"""
validate_layer4_embeddings.py — Layer 4 Embedding Unit Validator Entry Point

Runs the IdentityStore unit tests (no video source needed, no InsightFace
model download required for the pure-FAISS tests).

Usage:
    python validate_layer4_embeddings.py

Exit code 0 = all checks pass, 1 = one or more failures.
"""

import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.layer4_identity.embedding_validator import (
    run_embedding_shape_tests,
    run_embedding_unit_tests,
    print_summary,
    _fail_count
)

if __name__ == "__main__":
    print(f"\n{'='*58}")
    print(f"  Layer 4 — Embedding + IdentityStore Unit Validator")
    print(f"{'='*58}")
    run_embedding_shape_tests()
    run_embedding_unit_tests()
    print_summary()
    sys.exit(0 if _fail_count == 0 else 1)
