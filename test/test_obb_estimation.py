import numpy as np
import pytest
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from torch_obb import obb_estimate


def compute_obb_volume(vertices):
    """Compute the volume of an oriented bounding box defined by 8 vertices."""
    # Use the fact that OBB is a rectangular box
    # Compute the volume as the product of the extents along the three axes

    # Find the center
    center = np.mean(vertices, axis=0)

    # Find the three principal axes by looking at the vertex distribution
    # For an OBB, vertices should form a rectangular box
    # We can compute the covariance of the vertices to find the axes
    cov = np.cov((vertices - center).T)
    eigenvalues = np.linalg.eigvals(cov)

    # Volume is product of extents (sqrt of eigenvalues gives us the half-extents)
    # Actually, for a box with vertices at +/- half_sizes along each axis,
    # the volume is (2*half_size_x) * (2*half_size_y) * (2*half_size_z) = 8 * product(half_sizes)
    # And eigenvalues are approximately half_sizes^2, so sqrt(eigenvalues) ≈ half_sizes
    half_sizes = np.sqrt(np.abs(eigenvalues))
    volume = 8 * np.prod(half_sizes)

    return volume


def centers_rotations_halfsizes_to_vertices(centers, rotations, halfsizes):
    """Convert OBB representation (center, rotation, halfsizes) to 8 vertices.

    This is used to convert the output of the optimized Warp implementation
    to the vertex format expected by the tests.
    """
    if torch is None:
        raise ImportError("PyTorch is required for this conversion")

    # Convert inputs to torch tensors if needed
    centers = torch.from_numpy(np.asarray(centers)).float()
    rotations = torch.from_numpy(np.asarray(rotations)).float()
    halfsizes = torch.from_numpy(np.asarray(halfsizes)).float()

    # AABB corners in local coordinate system
    aabb_corners = torch.tensor([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ], dtype=torch.float32)

    # Scale by half-sizes
    scaled_corners = aabb_corners * halfsizes.unsqueeze(0)

    # Rotate and translate to world space
    # rotations is a matrix where columns are the principal axes
    obb_vertices = torch.matmul(scaled_corners, rotations.t()) + centers.unsqueeze(0)

    return obb_vertices.numpy()


def test_warp_simple_cube():
    """Test Warp implementation on a simple cube."""
    # Create a simple cube centered at origin
    vertices = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float32)

    # Test PyTorch baseline
    obb_baseline = oriented_bounding_box_pca_torch(vertices)

    # Should return 8 vertices
    assert obb_baseline.shape == (8, 3)

    # Should produce a valid bounding box (positive volume)
    # For a cube, the OBB should match the AABB
    expected_bounds = np.array([-1, 1])  # min and max bounds

    # Check that all baseline vertices are within expected bounds (with some tolerance)
    for i in range(3):  # x, y, z coordinates
        coords = obb_baseline[:, i]
        assert np.all(coords >= expected_bounds[0] - 1e-5)
        assert np.all(coords <= expected_bounds[1] + 1e-5)

    # All input vertices should be within the OBB bounds
    for v in vertices:
        for i in range(3):
            assert expected_bounds[0] - 1e-5 <= v[i] <= expected_bounds[1] + 1e-5

    # Test Warp implementation if available
    if WARP_AVAILABLE:
        centers, rotations, halfsizes = obb_estimate_warp(vertices)
        obb_warp = centers_rotations_halfsizes_to_vertices(centers, rotations, halfsizes)

        # Should return 8 vertices
        assert obb_warp.shape == (8, 3)

        # Compare with baseline - should be very close for this simple case
        # (allowing for ordering differences in vertices)
        def compare_obb_vertices(obb1, obb2, tolerance=1e-4):
            """Compare two sets of OBB vertices, allowing for different orderings."""
            # Check that all points in obb1 are close to some point in obb2
            for v1 in obb1:
                distances = np.linalg.norm(obb2 - v1, axis=1)
                assert np.min(distances) < tolerance, f"Vertex {v1} not found in second OBB"

            # Check volumes are similar (as a sanity check)
            vol1 = compute_obb_volume(obb1)
            vol2 = compute_obb_volume(obb2)
            assert abs(vol1 - vol2) < tolerance, f"Volumes differ: {vol1} vs {vol2}"

        compare_obb_vertices(obb_baseline, obb_warp)


def test_warp_random_points():
    """Test implementations on random point cloud."""
    np.random.seed(42)  # For reproducible results

    # Create random point cloud
    vertices = np.random.randn(100, 3).astype(np.float32)

    # Test PyTorch baseline
    obb_baseline = oriented_bounding_box_pca_torch(vertices)

    # Should return 8 vertices
    assert obb_baseline.shape == (8, 3)

    # Should produce a valid bounding box that contains all input points
    for i in range(3):  # x, y, z coordinates
        min_coord = np.min(obb_baseline[:, i])
        max_coord = np.max(obb_baseline[:, i])
        assert np.all(vertices[:, i] >= min_coord - 1e-4)
        assert np.all(vertices[:, i] <= max_coord + 1e-4)

    # Test Warp implementation if available
    if WARP_AVAILABLE:
        centers, rotations, halfsizes = obb_estimate_warp(vertices)
        obb_warp = centers_rotations_halfsizes_to_vertices(centers, rotations, halfsizes)

        # Should return 8 vertices
        assert obb_warp.shape == (8, 3)

        # Should produce a valid bounding box that contains all input points
        for i in range(3):  # x, y, z coordinates
            min_coord = np.min(obb_warp[:, i])
            max_coord = np.max(obb_warp[:, i])
            assert np.all(vertices[:, i] >= min_coord - 1e-4)
            assert np.all(vertices[:, i] <= max_coord + 1e-4)


def test_warp_rotated_cube():
    """Test implementations on a rotated cube."""
    # Create a rotated cube
    # Start with axis-aligned cube
    vertices = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float32)

    # Apply 45-degree rotation around z-axis
    rotation_matrix = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
        [np.sin(np.pi/4), np.cos(np.pi/4), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    vertices_rotated = (rotation_matrix @ vertices.T).T

    # Test PyTorch baseline
    obb_baseline = oriented_bounding_box_pca_torch(vertices_rotated)

    # Should return 8 vertices
    assert obb_baseline.shape == (8, 3)

    # Should produce a valid bounding box that contains all input points
    for i in range(3):  # x, y, z coordinates
        min_coord = np.min(obb_baseline[:, i])
        max_coord = np.max(obb_baseline[:, i])
        assert np.all(vertices_rotated[:, i] >= min_coord - 1e-4)
        assert np.all(vertices_rotated[:, i] <= max_coord + 1e-4)

    # Test Warp implementation if available
    if WARP_AVAILABLE:
        centers, rotations, halfsizes = obb_estimate_warp(vertices_rotated)
        obb_warp = centers_rotations_halfsizes_to_vertices(centers, rotations, halfsizes)

        # Should return 8 vertices
        assert obb_warp.shape == (8, 3)

        # Should produce a valid bounding box that contains all input points
        for i in range(3):  # x, y, z coordinates
            min_coord = np.min(obb_warp[:, i])
            max_coord = np.max(obb_warp[:, i])
            assert np.all(vertices_rotated[:, i] >= min_coord - 1e-4)
            assert np.all(vertices_rotated[:, i] <= max_coord + 1e-4)


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_torch_compatibility():
    """Test that the implementations work with PyTorch tensors."""
    # Create test vertices as torch tensor
    vertices = torch.tensor([
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]
    ], dtype=torch.float32)

    # Test PyTorch baseline with torch input
    obb_baseline = oriented_bounding_box_pca_torch(vertices.numpy())

    # Should return numpy array
    assert isinstance(obb_baseline, np.ndarray)
    assert obb_baseline.shape == (8, 3)

    # Should produce a valid bounding box
    for i in range(3):
        coords = obb_baseline[:, i]
        assert np.all(coords >= -1 - 1e-4)
        assert np.all(coords <= 1 + 1e-4)

    # Test Warp implementation if available
    if WARP_AVAILABLE:
        centers, rotations, halfsizes = obb_estimate_warp(vertices)
        obb_warp = centers_rotations_halfsizes_to_vertices(centers, rotations, halfsizes)

        # Should return numpy array or torch tensor
        assert isinstance(obb_warp, (np.ndarray, torch.Tensor))
        assert obb_warp.shape == (8, 3)

        # Should produce a valid bounding box
        for i in range(3):
            coords = obb_warp[:, i]
            assert np.all(coords >= -1 - 1e-4)
            assert np.all(coords <= 1 + 1e-4)


def test_empty_input():
    """Test behavior with empty input."""
    # Empty vertices
    vertices_empty = np.empty((0, 3), dtype=np.float32)

    # Test PyTorch baseline
    obb_baseline = oriented_bounding_box_pca_torch(vertices_empty)

    # Should return zeros
    assert np.allclose(obb_baseline, 0)

    # Test Warp implementation if available
    if WARP_AVAILABLE:
        centers, rotations, halfsizes = obb_estimate_warp(vertices_empty)
        obb_warp = centers_rotations_halfsizes_to_vertices(centers, rotations, halfsizes)

        # Should return zeros
        assert np.allclose(obb_warp, 0)


def test_single_point():
    """Test behavior with single point."""
    # Single vertex
    vertices_single = np.array([[1, 2, 3]], dtype=np.float32)

    # Test PyTorch baseline
    obb_baseline = oriented_bounding_box_pca_torch(vertices_single)

    # Should return 8 vertices
    assert obb_baseline.shape == (8, 3)

    # All vertices should be the same point (the bounding box of a single point)
    assert np.allclose(obb_baseline, vertices_single[0])

    # Test Warp implementation if available
    if WARP_AVAILABLE:
        centers, rotations, halfsizes = obb_estimate_warp(vertices_single)
        obb_warp = centers_rotations_halfsizes_to_vertices(centers, rotations, halfsizes)

        # Should return 8 vertices
        assert obb_warp.shape == (8, 3)

        # All vertices should be the same point (the bounding box of a single point)
        assert np.allclose(obb_warp, vertices_single[0])


def test_pytorch_vs_warp_comparison():
    """Test that PyTorch baseline and Warp implementation produce similar results."""
    if not WARP_AVAILABLE:
        pytest.skip("Warp not available for comparison")

    np.random.seed(123)  # For reproducible results

    # Test on various point clouds
    test_cases = [
        # Simple cube
        np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ], dtype=np.float32),
        # Random points
        np.random.randn(50, 3).astype(np.float32),
        # Elongated shape
        np.random.randn(30, 3).astype(np.float32) * np.array([5, 1, 1]),
    ]

    for i, vertices in enumerate(test_cases):
        # Compute OBBs with both implementations
        obb_baseline = oriented_bounding_box_pca_torch(vertices)
        centers, rotations, halfsizes = obb_estimate_warp(vertices)
        obb_warp = centers_rotations_halfsizes_to_vertices(centers, rotations, halfsizes)

        # Compare volumes
        vol_baseline = compute_obb_volume(obb_baseline)
        vol_warp = compute_obb_volume(obb_warp)

        # Volumes should be very close (within 1% relative error)
        rel_error = abs(vol_baseline - vol_warp) / max(vol_baseline, vol_warp)
        assert rel_error < 0.01, f"Volume mismatch in test case {i}: {vol_baseline} vs {vol_warp}"

        # Both should contain all input points
        for impl_name, obb in [("baseline", obb_baseline), ("warp", obb_warp)]:
            for axis in range(3):
                min_coord = np.min(obb[:, axis])
                max_coord = np.max(obb[:, axis])
                assert np.all(vertices[:, axis] >= min_coord - 1e-3), f"{impl_name} OBB doesn't contain all points on axis {axis}"
                assert np.all(vertices[:, axis] <= max_coord + 1e-3), f"{impl_name} OBB doesn't contain all points on axis {axis}"


if __name__ == "__main__":
    # Run basic tests
    test_warp_simple_cube()
    test_warp_random_points()
    test_warp_rotated_cube()
    test_empty_input()
    test_single_point()

    if torch is not None:
        test_torch_compatibility()

    if WARP_AVAILABLE:
        test_pytorch_vs_warp_comparison()

    print("All tests passed!")
