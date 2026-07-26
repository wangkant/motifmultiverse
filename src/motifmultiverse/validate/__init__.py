"""validate stage -- see README.md in this directory for rule / failure / check."""
from __future__ import annotations

from motifmultiverse.schema import (
    PeakSplitManifest,
    SplitRole,
    build_peak_split_manifest,
    peak_split_manifest_checksum,
)

from .base import (
    AnalysisMode,
    CrossFitFold,
    DecisionSplitArtifact,
    ValidationSplitArtifact,
    assert_artifact_split_compatibility,
    assert_cross_fit_compatibility,
    assert_split_compatibility,
)

__all__ = [
    "AnalysisMode",
    "CrossFitFold",
    "DecisionSplitArtifact",
    "PeakSplitManifest",
    "SplitRole",
    "ValidationSplitArtifact",
    "assert_artifact_split_compatibility",
    "assert_cross_fit_compatibility",
    "assert_split_compatibility",
    "build_peak_split_manifest",
    "peak_split_manifest_checksum",
    "run",
]


def run(*args, **kwargs):
    """Not implemented in the pre-alpha skeleton."""
    raise NotImplementedError(
        "validate is a skeleton; see src/motifmultiverse/validate/README.md"
    )
