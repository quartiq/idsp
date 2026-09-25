import numpy as np
import pytest
from idsp import atan2, cossin, sos


def test_trig_roundtrip():
    phase = np.array([0, 1 << 29, -(1 << 30), (1 << 31) - 1], dtype=np.int32)
    actual = atan2(cossin(phase))
    error = (actual.astype(np.int64) - phase + (1 << 31)) % (1 << 32) - (1 << 31)
    assert np.max(np.abs(error)) < 8192


@pytest.mark.parametrize("shape", [(2, 3), (3, 1), (1, 5), (0, 3)])
def test_coordinate_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        atan2(np.zeros(shape, dtype=np.int32))


def test_coordinate_layout():
    coordinates = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    for array in [coordinates[::2], coordinates[::-1], np.asfortranarray(coordinates)]:
        with pytest.raises(ValueError, match="C-contiguous"):
            atan2(array)
    with pytest.raises(ValueError, match="C-contiguous"):
        atan2(np.ones((2, 3), dtype=np.int32).T)


def test_empty_coordinates():
    assert atan2(np.empty((0, 2), dtype=np.int32)).shape == (0,)


def test_sos_identity():
    samples = np.array([1, -2, 3, -(1 << 30)], dtype=np.int32)
    expected = samples.copy()
    sos(np.array([[1.0, 0, 0, 1, 0, 0]]), samples)
    np.testing.assert_array_equal(samples, expected)
