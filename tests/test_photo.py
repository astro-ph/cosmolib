"""Tests for `cosmolib.data.photo`, in particular that its dataclasses
(`AngularPowerSpectrum`, `TwoPointCorrelationFunction`, `COSEBI`) don't
break `jax.grad` when constructed from a JAX array or tracer (see the
`26-fix-jax-clash-with-cloelib-photo-classes` branch this file belongs
to). `jax` is an optional, soft dependency - these tests are skipped
if it isn't installed.
"""

import importlib.util

import numpy as np
import pytest

from cosmolib.data.photo import AngularPowerSpectrum, COSEBI, TwoPointCorrelationFunction

_JAX_INSTALLED = importlib.util.find_spec("jax") is not None
if _JAX_INSTALLED:
    import jax
    import jax.numpy as jnp


@pytest.mark.parametrize("cls,extra", [
    (AngularPowerSpectrum, {"ell": np.array([1.0, 2.0, 3.0])}),
    (TwoPointCorrelationFunction, {"theta": np.array([1.0, 2.0, 3.0])}),
    (COSEBI, {}),
])
def test_plain_numpy_input_unaffected(cls, extra):
    """The non-JAX path must behave exactly as before: a plain float
    ndarray, regardless of jax's availability."""
    result = cls(array=[1, 2, 3], **extra)
    assert isinstance(result.array, np.ndarray)
    assert result.array.dtype == np.float64
    np.testing.assert_array_equal(result.array, [1.0, 2.0, 3.0])


@pytest.mark.skipif(not _JAX_INSTALLED, reason="jax not installed")
@pytest.mark.parametrize("cls,extra", [
    (AngularPowerSpectrum, {"ell": None}),
    (TwoPointCorrelationFunction, {"theta": None}),
    (COSEBI, {}),
])
def test_jax_array_input_stays_a_jax_array(cls, extra):
    """A concrete JAX array input must come back as a JAX array (not be
    silently downcast to NumPy), so later JAX ops on `.array` stay fast
    and device-resident."""
    result = cls(array=jnp.array([1.0, 2.0, 3.0]), **extra)
    assert isinstance(result.array, jax.Array)


@pytest.mark.skipif(not _JAX_INSTALLED, reason="jax not installed")
def test_angular_power_spectrum_is_differentiable_through_jax_grad():
    """The actual bug this branch fixes: constructing an
    `AngularPowerSpectrum` from a JAX tracer used to raise
    `jax.errors.TracerArrayConversionError` (a plain `np.asarray` call
    can't pull a concrete value out of an abstract trace) - this must
    now differentiate cleanly end-to-end.
    """

    def loss(x):
        aps = AngularPowerSpectrum(
            array=jnp.array([x, x**2, x**3]), ell=jnp.array([1.0, 2.0, 3.0])
        )
        return jnp.sum(aps.array)

    assert loss(2.0) == pytest.approx(14.0)
    grad = jax.grad(loss)(2.0)
    assert grad == pytest.approx(17.0)  # d/dx (x + x^2 + x^3) = 1 + 2x + 3x^2


@pytest.mark.skipif(not _JAX_INSTALLED, reason="jax not installed")
def test_jax_grad_through_jit_traced_construction():
    """Same check, but under `jax.jit` too - the construction happens
    with an abstract tracer (no concrete value at all), not just inside
    an un-jitted `jax.grad` call.
    """

    @jax.jit
    def loss(x):
        aps = AngularPowerSpectrum(array=x * jnp.array([1.0, 2.0, 3.0]))
        return jnp.sum(aps.array)

    grad = jax.grad(loss)(2.0)
    assert grad == pytest.approx(6.0)  # d/dx sum(x * [1,2,3]) = 1+2+3
