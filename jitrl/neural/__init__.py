"""Optional local Neural Lens; importing this package never imports PyTorch.

The remote language model remains frozen. Neural parameters, episodic facts
and reward settlements are three separate, versioned objects.
"""

from .config import LensConfig

__all__ = ["LensConfig"]
