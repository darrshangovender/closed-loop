"""F2 integration with the three sibling repositories (packets P11a to P11c).

This is the only package that may import the siblings or pandas, and only lazily: importing
``closed_loop.f2`` costs nothing, and a public function raises ``F2Unavailable`` when a sibling is
missing. Nothing here is imported by ``closed_loop`` itself.
"""

from closed_loop.f2._siblings import SIBLING_MODULES, F2Unavailable

__all__ = ["SIBLING_MODULES", "F2Unavailable"]
