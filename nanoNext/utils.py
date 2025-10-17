"""Import utilities for checking availability of optional dependencies."""

from __future__ import annotations


def is_causal_conv1d_available() -> bool:
    """Check if causal_conv1d package is available."""
    try:
        import causal_conv1d
        return True
    except ImportError:
        return False


def is_flash_linear_attention_available() -> bool:
    """Check if flash-linear-attention (fla) package is available."""
    try:
        import fla
        return True
    except ImportError:
        return False

