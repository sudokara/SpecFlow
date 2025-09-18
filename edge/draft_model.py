"""
Backward compatibility wrapper for EdgeDraftModel
DEPRECATED: Use edge.llm_draft_model.LLMDraftModel or edge.draft_model_factory instead
"""
import warnings
from edge.llm_draft_model import LLMDraftModel

# Issue deprecation warning
warnings.warn(
    "edge.draft_model.EdgeDraftModel is deprecated. "
    "Use edge.llm_draft_model.LLMDraftModel or edge.draft_model_factory.DraftModelFactory instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility alias
EdgeDraftModel = LLMDraftModel
