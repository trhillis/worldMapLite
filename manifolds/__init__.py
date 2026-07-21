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
]
