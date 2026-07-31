import numpy as np
from manifolds.mobius import FlatMobiusStrip

mobius = FlatMobiusStrip()

rng = np.random.default_rng(0)

# -------------------------
# Sample
# -------------------------

points = mobius.sample(10, rng=rng)

print("Points:")
print(points)

# -------------------------
# Distance
# -------------------------

distances = mobius.distance(
    points[:5],
    points[5:]
)

print("\nDistances:")
print(distances)

# -------------------------
# Distance matrix
# -------------------------

matrix = mobius.distance_matrix(points)

print("\nDistance matrix shape:")
print(matrix.shape)

print("\nSymmetric?")
print(np.allclose(matrix, matrix.T))

print("\nDiagonal zero?")
print(np.allclose(np.diag(matrix), 0))

# -------------------------
# Embedding
# -------------------------

embedded = mobius.embed(points)

print("\nEmbedded shape:")
print(embedded.shape)

print("\nEmbedded points:")
print(embedded)

p = np.array([[1.2, 0.3]])

g = np.array([
    [
        1.2 + mobius.length,
        mobius.width - 0.3,
    ]
])

print("\nDistance to glide image:")
print(mobius.distance(p, g))

print("\nEmbedding consistency:")

e1 = mobius.embed(p)
e2 = mobius.embed(g)

print(e1)
print(e2)

print(np.allclose(e1, e2))