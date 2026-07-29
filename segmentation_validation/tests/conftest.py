"""Make the sibling segmentation_validation modules importable during tests.

segmentation_validation is a workspace, not an installed package, so its modules
(``model_segmentation``, ``segmentation_dataset``, ``train_segmentation``) are
imported by top-level name. Add the workspace root to ``sys.path`` so the tests
resolve them the same way ``python train_segmentation.py`` does from that dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
