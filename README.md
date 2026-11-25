# **cosmolib**

_A lightweight Python library for handling cosmology data formats and creating publication-quality plots._
Designed for **clarity**, **composability**, and **seamless data format integration** into scientific cosmological softwares.

---

## 📘 Table of Contents

- [Installation](#installation)
- [License](#license)

---

## ⚙️ Installation

**From PyPI (recommended):**

```bash
pip install cosmolib
```

**From source (for developers and contributors):**

```bash
git clone https://github.com/astro-ph/cosmolib.git
cd cosmolib
pip install -e .
```

## 🧮 Mixing full-sky spectra

Given full-sky angular power spectra and mixing matrices in the cosmolib format
(dictionaries of `AngularPowerSpectrum` keyed by `(tracer1, tracer2, bin1, bin2)`,
e.g. as read with `euclidlib`), the pseudo-spectra are obtained with:

```python
from cosmolib.data import apply_mixing_matrices

pseudo_cls = apply_mixing_matrices(full_sky_cls, mixing_matrices)
```

Full-sky spectra must be unbinned (integer `ell` in unit steps; if they start
above `ell = 0` the missing multipoles are zero-filled). The output carries the
`ell`, `lower`, `upper` and `weight` of the mixing matrices. Spin-2 x spin-2
pairs use the `[M_EE->EE, M_BB->EE, M_EB->EB]` mixing-matrix stack. Use
`apply_mixing_matrix(cl, mm)` for a single pair. Both work with JAX arrays and
are differentiable.

## 📄 License

cosmolib is released under the MIT License.
You are free to use, modify, and distribute it under the terms of this license.
