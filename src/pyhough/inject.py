import numpy as np
import pyfstat
from dataclasses import dataclass
from typing import Callable, Tuple


def phase_from_frequency(tt,inject_fs):
    """
    inject_fs : array of instantaneous frequencies [Hz]
    tt        : array of times [s], same length

    returns
    -------
    phase : array of phase [rad]
    """
    inject_fs = np.asarray(inject_fs, dtype=float)
    tt = np.asarray(tt, dtype=float)

    dt = np.hstack((0,np.diff(tt)))
    phase_cycles = np.cumsum(inject_fs * dt)
    phase = np.mod(phase_cycles, 1.0) * 2.0 * np.pi
    return phase


def inject_into_sft(tt,inject_fs,amps,sft,NORM):

    phase = phase_from_frequency(tt,inject_fs)
    
    sig_t = amps * np.exp(1j * phase) ## need to include antenna pattern
    sig_f =  np.fft.fft(sig_t) * NORM
    sft = sft + sig_f
    return sft

def inject_sig_into_sft(sig_t,sft,NORM):

    sig_f =  np.fft.fft(sig_t) * NORM
    sft = sft + sig_f
    return sft


def calc_Hplus(eta,psi):
#     eta=simsour.eta; #both in radians
#     psi=simsour.psi*pi/180;
    Hp = np.sqrt(1 / (1 + eta**2)) * (np.cos(2 * psi) - 1j * eta * np.sin(2 * psi))
    return Hp

def calc_Hcross(eta,psi):
    Hc = np.sqrt(1 / (1 + eta**2)) * (np.sin(2 * psi) + 1j * eta * np.cos(2 * psi))
    
    return Hc


# -----------------------------
# Data containers (degrees in, radians internal)
# -----------------------------

import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Source:
    # --- defined parameters ---
    alpha_deg: float
    delta_deg: float
    psi_deg: float
    cosiota: float  # fundamental polarization parameter

    # --- derived parameters ---

    @property
    def eta(self) -> float:
        return -2 * self.cosiota / (1 + self.cosiota**2)

    def Hp(self):
        return calc_Hplus(
            self.eta,
            np.deg2rad(self.psi_deg)
        )
    def Hc(self):
        return calc_Hcross(
            self.eta,
            np.deg2rad(self.psi_deg)
        )
   

@dataclass(frozen=True)
class Antenna:
    name: str
    lat_deg: float
    long_deg: float
    azim_deg: float
    height_m: float
    winter_hour: float
    summer_hour: float
    incl: float
    type: int

def ligoh() -> Antenna:
    return Antenna(
        name="ligoh",
        lat_deg=46.455,
        long_deg=240.592,
        azim_deg=144.0006,
        height_m=142.5,
        winter_hour=-8,
        summer_hour=-7,
        incl=0,
        type=2,
    )

def ligol() -> Antenna:
    return Antenna(
        name="ligol",
        lat_deg=30.563,
        long_deg=269.266,
        azim_deg=72.2836,
        height_m=-6.5,
        winter_hour=-6,
        summer_hour=-5,
        incl=0,
        type=2,
    )


# -----------------------------
# Core: MATLAB sour_ant_2_5vec
# -----------------------------

def _sour_ant_2_5vec(alpha,delta,eta,psi_deg, ant: Antenna, culm: int = 0):
    """
    Faithful refactor of MATLAB sour_ant_2_5vec.
    Returns:
        L0 (A), L45 (B), CL, CR, v, Hp, Hc
    """
    # Convert degrees -> radians
    a   = np.deg2rad(alpha)
    d   = np.deg2rad(delta)
    eta = float(eta)
    psi = np.deg2rad(psi_deg)

    lat  = np.deg2rad(ant.lat_deg)
    lon  = np.deg2rad(ant.long_deg)
    az   = np.deg2rad(ant.azim_deg)

    if culm > 0:
        a = 0.0
        lon = 0.0

    al = np.exp(-1j * (a - lon))

    c2a, s2a = np.cos(2 * az), np.sin(2 * az)
    c2d, s2d = np.cos(2 * d),  np.sin(2 * d)
    c2l, s2l = np.cos(2 * lat), np.sin(2 * lat)

    cd, sd = np.cos(d), np.sin(d)
    cl, sl = np.cos(lat), np.sin(lat)

    # Coefficients (unchanged)
    a0  = -(3/16) * (1 + c2d) * (1 + c2l) * c2a
    a1c = -(1/4)  * s2d * s2l * c2a
    a1s = -(1/2)  * s2d * cl  * s2a
    a2c = -(1/16) * (3 - c2d) * (3 - c2l) * c2a
    a2s = -(1/4)  * (3 - c2d) * sl * s2a

    b1c = -cd * cl * s2a
    b1s = (1/2) * cd * s2l * c2a
    b2c = -sd * sl * s2a
    b2s = (1/4) * sd * (3 - c2l) * c2a

    A = np.empty(5, dtype=complex)
    B = np.empty(5, dtype=complex)

    A[0] = (al**-2) * (a2c + 1j * a2s) / 2
    A[1] = (al**-1) * (a1c + 1j * a1s) / 2
    A[2] = a0
    A[3] = (al)     * (a1c - 1j * a1s) / 2
    A[4] = (al**2)  * (a2c - 1j * a2s) / 2

    B[0] = (al**-2) * (b2c + 1j * b2s) / 2
    B[1] = (al**-1) * (b1c + 1j * b1s) / 2
    B[2] = 0.0
    B[3] = (al)     * (b1c - 1j * b1s) / 2
    B[4] = (al**2)  * (b2c - 1j * b2s) / 2

    L0, L45 = A, B

    # Circular 5-vectors (as in MATLAB: normalize A and B to equal total power)
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    if normA == 0 or normB == 0:
        raise ValueError("Degenerate antenna response: norm(A) or norm(B) is zero.")

    scale = np.sqrt(normA**2 + normB**2)
    A1 = scale * A / normA
    B1 = scale * B / normB

    CL = (A1 + 1j * B1) / np.sqrt(2)
    CR = (A1 - 1j * B1) / np.sqrt(2)

    # Polarization weights

    Hp = calc_Hplus(eta,psi)
    Hc = calc_Hcross(eta,psi)
    v = Hp * L0 + Hc * L45

    return L0, L45, CL, CR, v, Hp, Hc


# -----------------------------
# Core: MATLAB sim_ps_st
# -----------------------------

def sidereal_lf_series(alpha,delta,eta,psi_deg, ant: Antenna, N: int) -> np.ndarray:
    """
    Equivalent of MATLAB sim_ps_st's 'lf' output.
    Returns lf shape (N,), complex.
    """
    if N <= 0:
        raise ValueError("N must be positive.")

    st = (2 * np.pi / N) * np.arange(N, dtype=float)  # 0..2pi*(N-1)/N
    _, _, _, _, A, _, _ = _sour_ant_2_5vec(alpha,delta,eta,psi_deg, ant)


     # k = [-2, -1, 0, 1, 2] like ((1:5)-3) in MATLAB
    k = (np.arange(1, 6) - 3).astype(float)[:, None]   # (5,1)

    lf = np.sum(A[:, None] * np.exp(1j * k * st[None, :]), axis=0)
    return lf


# -----------------------------
# Indexing: reproduce MATLAB i1
# -----------------------------

def sidereal_bin_indices(
    st_hours: np.ndarray,
    nsid: int,
) -> np.ndarray:
    """
    MATLAB: i1 = mod(round(st*(nsid-1)/24), nsid-1) + 1
    Python returns 0-based indices in [0, nsid-2].
    """
    if nsid < 2:
        raise ValueError("nsid must be >= 2 (since code uses nsid-1).")

    bins = nsid - 1
    idx = np.mod(np.rint(st_hours * bins / 24.0).astype(np.int64), bins)
    return idx

def random_sky_deg():
    alpha = np.random.uniform(0.0, 360.0)            # degrees
    u = np.random.uniform(-1.0, 1.0)
    delta = np.degrees(np.arcsin(u))                 # degrees
    return alpha, delta

def random_cosiota():
    """
    Draw cosiota parameter eta uniformly in [-1, 1].

    Returns
    -------
    float
        cosiota in [-1, 1]
    """
    return np.random.uniform(-1.0, 1.0)


def random_psi():
    """
    Draw polarization angle psi uniformly in [-90, 90).

    Returns
    -------
    float
        psi in radians
    """
    return np.random.uniform(-90, 90)


def get_detector_velocities(gps_time_arr,tfft,ifo):



    mds = pyfstat.DetectorStates().get_multi_detector_states(
        timestamps=gps_time_arr,
        Tsft=tfft,
        detectors=ifo,
    )

    det_series = mds.data[0]
    states = det_series.data

    N = len(states)
    vs = np.empty((N, 3))

    for i, state in enumerate(states):
        vs[i, :] = state.vDetector.data
    return vs


