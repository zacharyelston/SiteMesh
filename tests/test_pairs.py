import numpy as np

from sitemesh.sfm import spatial_pairs


def _line(n):
    return {f"im{i}": np.array([float(i), 0.0, 0.0]) for i in range(n)}


def test_k_nearest_on_a_line():
    pos = _line(6)
    pairs = spatial_pairs(sorted(pos), pos, k=1)
    # each image pairs with its nearest neighbour only; ties break toward the lower index
    assert ("im0", "im1") in pairs
    assert ("im4", "im5") in pairs
    assert all(abs(int(a[2:]) - int(b[2:])) == 1 for a, b in pairs)


def test_vertical_axis_counts():
    # ARKit's vertical is Y: two shots stacked 2 m apart must not look co-located.
    pos = {"a": np.zeros(3), "b": np.array([0.0, 2.0, 0.0]), "c": np.array([0.1, 0.0, 0.0])}
    assert spatial_pairs(["a", "b", "c"], pos, k=1) == {("a", "c"), ("a", "b")}


def test_images_without_prior_pair_with_everything():
    pos = _line(4)
    names = sorted(pos) + ["limited"]
    pairs = spatial_pairs(names, pos, k=1)
    for n in pos:
        assert tuple(sorted((n, "limited"))) in pairs


def test_max_distance_and_k_capped():
    pos = {"a": np.zeros(3), "b": np.array([1000.0, 0, 0])}
    assert spatial_pairs(["a", "b"], pos, k=40, max_distance_m=500.0) == set()
    assert spatial_pairs(["a", "b"], pos, k=40, max_distance_m=2000.0) == {("a", "b")}
