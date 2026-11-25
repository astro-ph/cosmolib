"""Apply mixing matrices to full-sky angular power spectra.

Both inputs are in the cosmolib format, i.e. dictionaries of
:class:`~cosmolib.data.AngularPowerSpectrum` objects keyed by
``(tracer1, tracer2, bin1, bin2)`` tuples such as ``("POS", "SHE", 1, 2)``,
as produced by heracles and read by ``euclidlib.le3.pk_wl``.

Conventions
-----------
Full-sky spectra are unbinned, with the multipole axis last (or given by the
``axis`` attribute) and with integer ``ell`` running in unit steps.  Their
array shape encodes the spin of the pair:

* spin-0 x spin-0 (e.g. ``POS``-``POS``): ``(n_ell,)``
* spin-0 x spin-2 (e.g. ``POS``-``SHE``): ``(2, n_ell)`` for ``(E, B)``
* spin-2 x spin-2 (e.g. ``SHE``-``SHE``): ``(2, 2, n_ell)`` for
  ``[[EE, EB], [BE, BB]]``

Mixing matrices have the *output* multipole axis second to last (this is the
``axis`` heracles sets) and the *input* multipole axis last, with input
multipoles ``ell = 0, 1, ..., ell_max``.  Their shape is

* ``(n_out, n_in)`` if either field has spin 0 -- the same matrix acts on
  every E/B component;
* ``(3, n_out, n_in)`` for spin-2 x spin-2, stacking
  ``[M_EE->EE, M_BB->EE, M_EB->EB]``, such that ::

      EE' = M0 @ EE + M1 @ BB
      BB' = M0 @ BB + M1 @ EE
      EB' = M2 @ EB,   BE' = M2 @ BE

The output (pseudo-)spectra carry the ``ell``, ``lower``, ``upper`` and
``weight`` of the mixing matrix, so they can be compared directly to measured,
possibly binned, pseudo-spectra.

The routines work on NumPy arrays and, when ``jax`` is installed and the
inputs are JAX arrays or tracers, stay differentiable under ``jax.grad`` and
``jax.jit``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .photo import AngularPowerSpectrum, _jax

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType
    from typing import Any

    from numpy.typing import NDArray

__all__ = [
    "apply_mixing_matrix",
    "apply_mixing_matrices",
]


def _array_namespace(*arrays: Any) -> ModuleType:
    """Return ``jax.numpy`` if any input is a JAX array, else ``numpy``."""
    if _jax is not None and any(isinstance(a, _jax.Array) for a in arrays):
        return _jax.numpy
    return np


def _full_sky_array(cl: AngularPowerSpectrum, n_in: int, xp: ModuleType) -> NDArray[Any]:
    """Return the array of a full-sky spectrum on ``ell = 0, ..., n_in - 1``.

    The multipole axis is moved last.  If the spectrum starts at ``ell > 0``
    the missing low multipoles are filled with zeros; multipoles beyond the
    range of the mixing matrix are discarded.
    """
    if len(cl.axis) != 1:
        raise ValueError(
            f"full-sky spectrum must have a single ell axis, got axis={cl.axis}"
        )
    arr = xp.moveaxis(cl.array, cl.axis[0], -1)

    start = 0
    if cl.ell is not None:
        if isinstance(cl.ell, tuple):
            raise ValueError("full-sky spectrum must have a single ell array")
        ell = np.asarray(cl.ell)
        if ell.shape != (arr.shape[-1],):
            raise ValueError(
                f"ell has shape {ell.shape}, but the spectrum has "
                f"{arr.shape[-1]} multipoles"
            )
        if ell.size > 0:
            if np.any(ell != np.round(ell)) or np.any(np.diff(ell) != 1):
                raise ValueError(
                    "full-sky spectrum must be unbinned, with integer ell in "
                    "steps of one"
                )
            start = int(ell[0])
            if start < 0:
                raise ValueError("full-sky spectrum has negative ell")

    if start > 0:
        pad = xp.zeros(arr.shape[:-1] + (start,), dtype=arr.dtype)
        arr = xp.concatenate([pad, arr], axis=-1)

    if arr.shape[-1] < n_in:
        raise ValueError(
            f"mixing matrix needs the full-sky spectrum up to ell = {n_in - 1}, "
            f"but it only goes up to ell = {arr.shape[-1] - 1}"
        )
    return arr[..., :n_in]


def apply_mixing_matrix(
    cl: AngularPowerSpectrum,
    mm: AngularPowerSpectrum,
) -> AngularPowerSpectrum:
    """Multiply a single full-sky spectrum by its mixing matrix.

    Parameters
    ----------
    cl : AngularPowerSpectrum
        Full-sky angular power spectrum of one tracer/bin pair (see the module
        docstring for the expected shapes).
    mm : AngularPowerSpectrum
        Mixing matrix of the same tracer/bin pair.

    Returns
    -------
    AngularPowerSpectrum
        Mixed (pseudo-)spectrum, with the ``ell``, ``lower``, ``upper`` and
        ``weight`` of the mixing matrix.
    """
    xp = _array_namespace(cl.array, mm.array)
    M = mm.array
    if M.ndim not in (2, 3):
        raise ValueError(
            f"mixing matrix must have shape (n_out, n_in) or (3, n_out, n_in), "
            f"got {M.shape}"
        )
    n_in = M.shape[-1]
    C = _full_sky_array(cl, n_in, xp)

    if C.ndim in (1, 2):
        # spin-0 x spin-0, or spin-0 x spin-2 where the same matrix acts on
        # both the E and the B component
        if M.ndim != 2:
            raise ValueError(
                f"spectrum of shape {cl.shape} needs a mixing matrix of shape "
                f"(n_out, n_in), got {M.shape}"
            )
        if C.ndim == 2 and C.shape[0] != 2:
            raise ValueError(
                f"spin-0 x spin-2 spectrum must have shape (2, n_ell), got {cl.shape}"
            )
        out = xp.einsum("ij,...j->...i", M, C)

    elif C.ndim == 3:
        # spin-2 x spin-2: E/B mixing
        if C.shape[:2] != (2, 2):
            raise ValueError(
                f"spin-2 x spin-2 spectrum must have shape (2, 2, n_ell), "
                f"got {cl.shape}"
            )
        if M.ndim != 3 or M.shape[0] != 3:
            raise ValueError(
                f"spectrum of shape {cl.shape} needs a mixing matrix of shape "
                f"(3, n_out, n_in), got {M.shape}"
            )
        EE, EB, BE, BB = C[0, 0], C[0, 1], C[1, 0], C[1, 1]
        out_EE = M[0] @ EE + M[1] @ BB
        out_BB = M[0] @ BB + M[1] @ EE
        out_EB = M[2] @ EB
        out_BE = M[2] @ BE
        out = xp.stack([xp.stack([out_EE, out_EB]), xp.stack([out_BE, out_BB])])

    else:
        raise ValueError(f"unsupported full-sky spectrum shape {cl.shape}")

    return AngularPowerSpectrum(
        array=out,
        ell=mm.ell,
        axis=out.ndim - 1,
        lower=mm.lower,
        upper=mm.upper,
        weight=mm.weight,
        software=cl.software,
        metadata=cl.metadata,
    )


def _swap(cl: AngularPowerSpectrum) -> AngularPowerSpectrum:
    """Return the spectrum of the swapped pair ``(b, a)`` given ``(a, b)``."""
    if cl.ndim == 3 and len(cl.axis) == 1 and cl.axis[0] == 2:
        # [[EE, EB], [BE, BB]] -> [[EE, BE], [EB, BB]]
        xp = _array_namespace(cl.array)
        return replace(cl, array=xp.swapaxes(cl.array, 0, 1), axis=cl.axis)
    return cl


def apply_mixing_matrices(
    cls: Mapping[Any, AngularPowerSpectrum],
    mms: Mapping[Any, AngularPowerSpectrum],
) -> dict[Any, AngularPowerSpectrum]:
    """Multiply full-sky spectra by their mixing matrices.

    Parameters
    ----------
    cls : mapping
        Full-sky angular power spectra in the cosmolib format, keyed by
        ``(tracer1, tracer2, bin1, bin2)``.
    mms : mapping
        Mixing matrices in the cosmolib format, with the same kind of keys.

    Returns
    -------
    dict
        Mixed (pseudo-)spectra, one for each key of ``mms``.

    Notes
    -----
    Every key of ``mms`` is computed.  If a key ``(a, b, i, j)`` is not in
    ``cls``, the swapped key ``(b, a, j, i)`` is used instead (transposing
    the E/B components for spin-2 x spin-2 spectra); if neither is present a
    :class:`KeyError` is raised.

    Examples
    --------
    >>> from euclidlib.le3 import pk_wl  # doctest: +SKIP
    >>> cls = pk_wl.angular_power_spectra("full_sky_cls.fits")  # doctest: +SKIP
    >>> mms = pk_wl.mixing_matrices("mixing_matrices.fits")  # doctest: +SKIP
    >>> pseudo_cls = apply_mixing_matrices(cls, mms)  # doctest: +SKIP
    """
    out = {}
    for key, mm in mms.items():
        if key in cls:
            cl = cls[key]
        else:
            swapped = (
                (key[1], key[0], key[3], key[2])
                if isinstance(key, tuple) and len(key) == 4
                else None
            )
            if swapped is None or swapped not in cls:
                raise KeyError(f"no full-sky spectrum for mixing matrix {key!r}")
            cl = _swap(cls[swapped])
        out[key] = apply_mixing_matrix(cl, mm)
    return out
