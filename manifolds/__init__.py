"""Manifold geodesic-distance exercise: interface stubs.

Each import line below is independent; adding a manifold to the roster is one
new module plus one line here, and nothing else changes. The polyhedra module
already carries both the octahedron and (for free, same machinery) the
icosahedron.
"""

from .base import Manifold
from .flat_torus import FlatTorus
from .mobius import FlatMobiusStrip
from .polyhedra import (
    PolyhedralSurface,
    octahedron,
    icosahedron,
    OCTAHEDRON_VERTICES,
    OCTAHEDRON_FACES,
    ICOSAHEDRON_VERTICES,
    ICOSAHEDRON_FACES,
)

__all__ = [
    "Manifold",
    "FlatTorus",
    "FlatMobiusStrip",
    "PolyhedralSurface",
    "octahedron",
    "icosahedron",
    "OCTAHEDRON_VERTICES",
    "OCTAHEDRON_FACES",
    "ICOSAHEDRON_VERTICES",
    "ICOSAHEDRON_FACES",
    "get_manifold",
]

# Name -> zero-arg factory, so a manifold instance can be reconstructed from
# a saved config/checkpoint string. Extend this dict as more manifolds are
# implemented; nothing else in the training/analysis pipeline needs to change.
_MANIFOLD_FACTORIES = {
    "octahedron": octahedron,
    "icosahedron": icosahedron,
    "flat_torus": FlatTorus,
    "mobius": FlatMobiusStrip,
}


def get_manifold(name: str) -> Manifold:
    """Look up a manifold instance by name (see _MANIFOLD_FACTORIES)."""

    if name not in _MANIFOLD_FACTORIES:
        raise ValueError(
            f"Unknown manifold {name!r}; available: "
            f"{sorted(_MANIFOLD_FACTORIES)}"
        )

    return _MANIFOLD_FACTORIES[name]()
