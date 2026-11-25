"""Tests for `cosmolib.data.mixing`: multiplying full-sky angular power
spectra by mixing matrices, both in the cosmolib format."""

import importlib.util

import numpy as np
import pytest

from cosmolib.data import (
    AngularPowerSpectrum,
    apply_mixing_matrices,
    apply_mixing_matrix,
)

_JAX_INSTALLED = importlib.util.find_spec("jax") is not None
if _JAX_INSTALLED:
    import jax
    import jax.numpy as jnp

N_IN = 20  # full-sky ell = 0, ..., 19
N_OUT = 6  # e.g. 6 output bandpowers


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _mm(rng, spin2x2=False):
    shape = (3, N_OUT, N_IN) if spin2x2 else (N_OUT, N_IN)
    edges = np.linspace(2, N_IN, N_OUT + 1)
    return AngularPowerSpectrum(
        array=rng.normal(size=shape),
        ell=0.5 * (edges[1:] + edges[:-1]),
        axis=len(shape) - 2,  # heracles convention: output ell axis
        lower=edges[:-1],
        upper=edges[1:],
        weight=np.ones(N_OUT),
    )


def _cl(rng, shape, ell=None):
    if ell is None:
        ell = np.arange(shape[-1])
    return AngularPowerSpectrum(array=rng.normal(size=shape), ell=ell)


def _reference(cl, M):
    """Literal copy of the formulas in cloelib's `get_pseudo_Cl`."""
    if cl.ndim == 1:
        return M @ cl
    if cl.ndim == 2:
        return np.stack([M @ cl[0], M @ cl[1]])
    arr = np.zeros((2, 2, M.shape[1]))
    arr[0, 0] = M[0] @ cl[0, 0] + M[1] @ cl[1, 1]
    arr[0, 1] = M[2] @ cl[0, 1]
    arr[1, 0] = M[2] @ cl[1, 0]
    arr[1, 1] = M[0] @ cl[1, 1] + M[1] @ cl[0, 0]
    return arr


@pytest.mark.parametrize(
    "shape,spin2x2",
    [((N_IN,), False), ((2, N_IN), False), ((2, 2, N_IN), True)],
    ids=["POS-POS", "POS-SHE", "SHE-SHE"],
)
def test_matches_cloelib_formulas(rng, shape, spin2x2):
    cl = _cl(rng, shape)
    mm = _mm(rng, spin2x2)
    out = apply_mixing_matrix(cl, mm)
    assert isinstance(out, AngularPowerSpectrum)
    assert out.shape == shape[:-1] + (N_OUT,)
    assert out.axis == (len(shape) - 1,)
    np.testing.assert_allclose(out.array, _reference(cl.array, mm.array))
    # binning information comes from the mixing matrix
    np.testing.assert_array_equal(out.ell, mm.ell)
    np.testing.assert_array_equal(out.lower, mm.lower)
    np.testing.assert_array_equal(out.upper, mm.upper)
    np.testing.assert_array_equal(out.weight, mm.weight)


def test_identity_mixing_matrix_is_a_no_op(rng):
    cl = _cl(rng, (2, 2, N_IN))
    eye = np.eye(N_IN)
    mm = AngularPowerSpectrum(
        array=np.stack([eye, np.zeros_like(eye), eye]), ell=np.arange(N_IN), axis=1
    )
    out = apply_mixing_matrix(cl, mm)
    np.testing.assert_allclose(out.array, cl.array)


def test_longer_full_sky_spectrum_is_truncated(rng):
    cl = _cl(rng, (N_IN + 15,))
    mm = _mm(rng)
    out = apply_mixing_matrix(cl, mm)
    np.testing.assert_allclose(out.array, mm.array @ cl.array[:N_IN])


def test_spectrum_starting_above_ell_zero_is_zero_padded(rng):
    cl_full = _cl(rng, (2, 2, N_IN))
    cl_full.array[..., :2] = 0.0  # spin-2 spectra vanish at ell < 2
    cl_from_2 = AngularPowerSpectrum(
        array=cl_full.array[..., 2:], ell=np.arange(2, N_IN)
    )
    mm = _mm(rng, spin2x2=True)
    np.testing.assert_allclose(
        apply_mixing_matrix(cl_from_2, mm).array,
        apply_mixing_matrix(cl_full, mm).array,
    )


def test_ell_axis_not_last(rng):
    cl = _cl(rng, (2, N_IN))
    cl_t = AngularPowerSpectrum(array=cl.array.T, ell=np.arange(N_IN), axis=0)
    mm = _mm(rng)
    np.testing.assert_allclose(
        apply_mixing_matrix(cl_t, mm).array, apply_mixing_matrix(cl, mm).array
    )


@pytest.mark.parametrize(
    "cl_shape,ell,spin2x2,match",
    [
        ((N_IN - 1,), None, False, "up to ell"),
        ((N_IN,), np.arange(0, 2 * N_IN, 2), False, "unbinned"),
        ((N_IN,), None, True, r"\(n_out, n_in\)"),
        ((2, 2, N_IN), None, False, r"\(3, n_out, n_in\)"),
        ((3, N_IN), None, False, r"\(2, n_ell\)"),
    ],
    ids=["too-short", "binned", "spin0-with-3d-mm", "spin2-with-2d-mm", "bad-shape"],
)
def test_errors(rng, cl_shape, ell, spin2x2, match):
    cl = _cl(rng, cl_shape, ell)
    with pytest.raises(ValueError, match=match):
        apply_mixing_matrix(cl, _mm(rng, spin2x2))


def test_apply_mixing_matrices_dict(rng):
    cls = {
        ("POS", "POS", 1, 1): _cl(rng, (N_IN,)),
        ("POS", "SHE", 1, 2): _cl(rng, (2, N_IN)),
        ("SHE", "SHE", 1, 2): _cl(rng, (2, 2, N_IN)),
        ("SHE", "SHE", 2, 2): _cl(rng, (2, 2, N_IN)),  # no mixing matrix
    }
    mms = {
        ("POS", "POS", 1, 1): _mm(rng),
        ("POS", "SHE", 1, 2): _mm(rng),
        ("SHE", "SHE", 1, 2): _mm(rng, spin2x2=True),
    }
    out = apply_mixing_matrices(cls, mms)
    assert set(out) == set(mms)
    for key in mms:
        np.testing.assert_allclose(
            out[key].array, _reference(cls[key].array, mms[key].array)
        )


def test_apply_mixing_matrices_uses_swapped_key(rng):
    she_pos = _cl(rng, (2, N_IN))
    she_she = _cl(rng, (2, 2, N_IN))
    cls = {("SHE", "POS", 2, 1): she_pos, ("SHE", "SHE", 2, 1): she_she}
    mms = {
        ("POS", "SHE", 1, 2): _mm(rng),
        ("SHE", "SHE", 1, 2): _mm(rng, spin2x2=True),
    }
    out = apply_mixing_matrices(cls, mms)
    np.testing.assert_allclose(
        out["POS", "SHE", 1, 2].array,
        _reference(she_pos.array, mms["POS", "SHE", 1, 2].array),
    )
    # (SHE_2, SHE_1) -> (SHE_1, SHE_2) swaps EB <-> BE
    np.testing.assert_allclose(
        out["SHE", "SHE", 1, 2].array,
        _reference(she_she.array.swapaxes(0, 1), mms["SHE", "SHE", 1, 2].array),
    )


def test_apply_mixing_matrices_missing_key(rng):
    with pytest.raises(KeyError, match="no full-sky spectrum"):
        apply_mixing_matrices({}, {("POS", "POS", 1, 1): _mm(rng)})


@pytest.mark.skipif(not _JAX_INSTALLED, reason="jax not installed")
@pytest.mark.parametrize(
    "shape,spin2x2",
    [((N_IN,), False), ((2, N_IN), False), ((2, 2, N_IN), True)],
    ids=["POS-POS", "POS-SHE", "SHE-SHE"],
)
def test_jax_grad_and_jit(rng, shape, spin2x2):
    mm = _mm(rng, spin2x2)
    base = rng.normal(size=shape)

    @jax.jit
    def loss(a):
        cl = AngularPowerSpectrum(array=a * jnp.asarray(base), ell=np.arange(N_IN))
        return jnp.sum(apply_mixing_matrix(cl, mm).array)

    expected = _reference(base, mm.array).sum()
    assert loss(1.0) == pytest.approx(expected, rel=1e-5)
    assert jax.grad(loss)(1.0) == pytest.approx(expected, rel=1e-5)
