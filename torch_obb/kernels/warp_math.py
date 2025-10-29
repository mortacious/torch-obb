import warp as wp

@wp.func
def norm3(vec: wp.vec3) -> wp.float32:
    """Compute squared norm of a 3D vector."""
    return wp.sqrt(wp.dot(vec, vec))

@wp.func
def normalize3(vec: wp.vec3, epsilon: wp.float32 = wp.float32(1e-6)) -> wp.vec3:
    """Normalize a 3D vector."""
    n = wp.sqrt(norm3(vec))
    if n < wp.float32(epsilon):
        n = wp.float32(epsilon)
    return vec / n