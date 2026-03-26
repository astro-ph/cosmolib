from __future__ import annotations
from dataclasses import dataclass

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Any, Union, Tuple
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PowerSpectrumMultipoles:
    """
    Output data model to store power spectrum Legendre multipole measurements.

    Attributes
    ----------
    k : ndarray
        Centers of $k$ bins
    keff : ndarray
        Effective values of $k$ bins
    Nmodes : ndarray
        Number of modes in each $k$ bin
    multipoles : ndarray
        Legendre multipoles of the anisotropic power spectrum
    fiducial_cosmology : dict[str, float]
        Fiducial cosmology used in the measurement
    zeff : ndarray
        Effective redshift of the sample
    nbar : float
        Number density of the sample
    Psn : float
        Poissonian shot noise contribution
    """
    k: NDArray[Any]
    keff: NDArray[Any]
    Nmodes: NDArray[Any]
    multipoles: NDArray[Any]
    fiducial_cosmology: dict[str, float]
    zeff: float
    nbar: float
    Psn: float

    def __post_init__(self) -> None:
        if any([len(self.k) != len(self.keff),
                len(self.k) != len(self.Nmodes),]):
            raise ValueError(
                "Inconsistent class attributes, all arrays must have the same "
                " length.")

    def __len__(self) -> int:
        return len(self.k)


@dataclass(frozen=True)
class TwoPointCorrelationCartesian:
    """
    Output data model to store 2-point correlation function (cartesian) measurements.

    Attributes
    ----------
    s_perp : ndarray
        Separation $s_\perp$ ortoghonal to the line of sight
    s_para : ndarray
        Separation $s_\parallel$ parallel to the line of sight
    correlation : ndarray
        2D cartesian 2PCF
    fiducial_cosmology : dict[str, float]
        Fiducial cosmology used in the measurement
    zeff : ndarray
        Effective redshift of the sample
    """
    s_perp: NDArray[Any]
    s_para: NDArray[Any]
    correlation: NDArray[Any]
    fiducial_cosmology: dict[str, float]
    zeff: float

    def __len__(self) -> int:
        return (len(self.s_perp), len(self.s_para))


@dataclass(frozen=True)
class TwoPointCorrelationPolar:
    """
    Output data model to store 2-point correlation function (polar) measurements.

    Attributes
    ----------
    s : ndarray
        Separation $s$
    mu : ndarray
        Cosinus $\mu$ of the angle to the line of sight
    correlation : ndarray
        2D polar 2PCF
    fiducial_cosmology : dict[str, float]
        Fiducial cosmology used in the measurement
    zeff : ndarray
        Effective redshift of the sample
    """
    s: NDArray[Any]
    mu: NDArray[Any]
    correlation: NDArray[Any]
    fiducial_cosmology: dict[str, float]
    zeff: float

    def __len__(self) -> int:
        return (len(self.s), len(self.mu))


@dataclass(frozen=True)
class TwoPointCorrelationMultipoles:
    """
    Output data model to store 2-point correlation function Legendre multipole measurements.

    Attributes
    ----------
    s : ndarray
        Separation $s$
    multipoles : ndarray
        Legendre multipoles of the anisotropic 2PCF
    fiducial_cosmology : dict[str, float]
        Fiducial cosmology used in the measurement
    zeff : ndarray
        Effective redshift of the sample
    """
    s: NDArray[Any]
    multipoles: NDArray[Any]
    fiducial_cosmology: dict[str, float]
    zeff: float

    def __len__(self) -> int:
        return len(self.s)


@dataclass(frozen=True)
class PowerSpectrumMultipolesCovariance:
    """
    Output data model to store covariance matrices of the power spectrum
    Legendre multipole measurements.

    Attributes
    ----------
    k : ndarray
        Centers of $k$ bins
    cov : dict
        Auto and cross covariances of the Legendre multipoles of the
        power spectrum
    zeff : ndarray
        Effective redshift of the sample
    """
    k: NDArray[Any]
    covariance: dict[str, NDArray[Any]]
    zeff: float

    def __len__(self) -> int:
        return len(self.k)


@dataclass(frozen=True)
class TwoPointCorrelationMultipolesCovariance:
    """
    Output data model to store covariance matrices of the 2-point correlation
    Legendre
    multipole measurements.

    Attributes
    ----------
    s : ndarray
        Separation $s$
    cov : dict
        Auto and cross covariances of the Legendre multipoles of the 2PCF
    zeff : ndarray
        Effective redshift of the sample
    """
    s: NDArray[Any]
    covariance: dict[str, NDArray[Any]]
    zeff: float

    def __len__(self) -> int:
        return len(self.s)


@dataclass(frozen=True)
class PowerSpectrumMultipolesMixingMatrix:
    """
    Output data model to store mixing matrices of the power spectrum Legendre
    multipole measurements.

    Attributes
    ----------
    kout : ndarray
        Output wavemodes $k$
    kin : dict[ndarray]
        Input wavemodes $k$ per multipole
    mixing : dict[ndarray]
        Individual blocks of the mixing matrix
    """
    kout: NDArray[Any]
    kin: dict[Any, NDArray[Any]]
    mixing: dict[str, NDArray[Any]]
    zeff: float

    def __len__(self) -> int:
        return (len(self.kout), len(self.kin))


@dataclass(frozen=True)
class BaryonAcousticOscillations:
    """
    Output data model to store alpha parameters from fits of the
    post-reconstruction spectroscopic observables.

    Attributes
    ----------
    alpha_par : float
        Value of parallel BAO scale parameter
    alpha_perp : float
        Value of perpendicular BAO scale parameter
    alpha_iso : float
        Value of isotropic BAO scale parameter
    alpha_ap : float
        Value of Alcock-Packzynski parameter
    fiducial_cosmology : dict[str, float]
        Fiducial cosmology used in the measurement
    zeff : float
        Effective redshift of the sample
    """
    alpha_par: float
    alpha_perp: float
    alpha_iso: float
    alpha_ap: float
    fiducial_cosmology: dict[str, float]
    zeff: float


@dataclass(frozen=True)
class BaryonAcousticOscillationsCovariance:
    """
    Output data model to store covariance for the BAO alpha parameters

    Attributes
    ----------
    observables : str or tuple of str
        Observables in the covariance, in the correct order
    covariance : dict
        Covariance of the BAO alphas
    correction_factor : float
        Correction factor m2 for the covariance, as in eq. of Percival+2014
    zeff : float
        Effective redshift of the sample
    """
    observables: Union[str, Tuple[str, ...]]
    covariance: dict[str, NDArray[Any]]
    correction_factor: float
    zeff: float
