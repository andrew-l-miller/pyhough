import numpy as np
from pyhough import physics,pm

def simulate_sinusoid(tt,f0,h0):
    freqs = f0 * np.ones((len(tt)))
    amps = h0 * np.ones((len(tt)))
    return amps,freqs


def simulate_sinusoid_with_drift(tt,f0,fdot,h0):
    freqs = f0 + fdot * tt
    amps = h0 * (freqs / f0)**2
    return amps,freqs

def simulate_cw(tt,f0,fdot,alpha,delta,h0,vs):
    
    _, freqs_no_dopp = simulate_sinusoid_with_drift(tt, f0, fdot, h0=0.0)  # dummy h0
    vec_n = pm.astro2rect([alpha,delta],1)
    freqs = freqs_no_dopp * (1 + np.dot(vec_n,vs.T))
    amps = h0 * (freqs_no_dopp / f0)**2
    return amps,freqs

def simulate_power_law(tt,f0,k,n,h0):
    const=k * (n-1) * f0**(n-1)
    
    if n == 1:
        fsss = f0 * np.exp(-k * tt)
    elif n == 11/3:
        fsss = f0 * (1 - const * tt)**(-1. / (n-1))
    else:
        fsss = f0 * (1 + const * tt)**(-1. / (n-1))
    
    if (n < 6.8) & (n != 11/3):
        amps = h0 * fsss**2 / f0**2 # ns
    elif (n >= 6.8) & (n < 7.2):
        amps = h0 * fsss**3 / f0**3 #rmode
    elif n == 11/3:
        amps = h0 * fsss**(2/3) / f0**(2/3) #binary

 
    return amps,fsss


def simulate_cbc_pn(tt, m1, m2, t_c, h0, order=3.5):
    """
    tt: absolute time array in GPS seconds (or any consistent absolute time base)
    t_c: coalescence time in same units as tt
    """
    tau = t_c - tt  # seconds-to-merger

    freqs, _, _ = cbc_calc_pn_freq(m1, m2, tau, order=order)

    amps = np.full_like(freqs, h0, dtype=float)  # placeholder amplitude
    return amps, freqs


def cbc_calc_pn_freq(m1, m2, tau, order=3.5):
    """

    Parameters
    ----------
    m1, m2 : float or array-like
        Component masses in solar masses.
    tau : float or array-like
        Seconds to merger (tc - t), same shape/broadcastable with m1, m2.
    order : float
        PN order in {1, 1.5, 2, 2.5, 3, 3.5}.
    consts : Constants
        Physical constants container.

    Returns
    -------
    fgw : ndarray
        PN GW frequency (Hz).
    fgw_0pn : ndarray
        Leading-order (0PN) GW frequency (Hz).
    df : ndarray
        abs(fgw - fgw_0pn) (Hz).
    """
    
    cc = physics.constants()
    c = cc['c']
    G = cc['G']
    Msun = cc['Msun']

    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    tau = np.asarray(tau, dtype=float)

    m = m1 + m2
    nu = (m1 * m2) / (m ** 2)  # symmetric mass ratio

    # theta = nu*c^3/(5*G*m*Msun) * tau
    theta = nu * c**3 / (5.0 * G * m * Msun) * tau

    C = 0.577215664901533  # Euler-Mascheroni gamma

    PN_0 = 1.0
    PN_1 = 0.0
    PN_1_5 = 0.0
    PN_2 = 0.0
    PN_2_5 = 0.0
    PN_3 = 0.0
    PN_3_5 = 0.0

    # Helper powers
    theta_m1_4 = theta ** (-1.0 / 4.0)
    theta_m3_8 = theta ** (-3.0 / 8.0)
    theta_m1_2 = theta ** (-1.0 / 2.0)
    theta_m5_8 = theta ** (-5.0 / 8.0)
    theta_m3_4 = theta ** (-3.0 / 4.0)
    theta_m7_8 = theta ** (-7.0 / 8.0)

    # MATLAB uses exact == comparisons on doubles; in Python be tolerant
    def _is(x, y, tol=1e-12):
        return abs(x - y) <= tol

    if _is(order, 1.0):
        PN_1 = (743/4042 + (11/48) * nu) * theta_m1_4

    elif _is(order, 1.5):
        PN_1 = (743/4042 + (11/48) * nu) * theta_m1_4
        PN_1_5 = -(1/5) * np.pi * theta_m3_8

    elif _is(order, 2.0):
        PN_1 = (743/4042 + (11/48) * nu) * theta_m1_4
        PN_1_5 = -(1/5) * np.pi * theta_m3_8
        PN_2 = (19583/254016 + (24401/193536) * nu + (31/288) * nu**2) * theta_m1_2

    elif _is(order, 2.5):
        PN_1 = (743/4042 + (11/48) * nu) * theta_m1_4
        PN_1_5 = -(1/5) * np.pi * theta_m3_8
        PN_2 = (19583/254016 + (24401/193536) * nu + (31/288) * nu**2) * theta_m1_2
        PN_2_5 = (-(11891/53760) + (109/1920) * nu) * np.pi * theta_m5_8

    elif _is(order, 3.0):
        PN_1 = (743/4042 + (11/48) * nu) * theta_m1_4
        PN_1_5 = -(1/5) * np.pi * theta_m3_8
        PN_2 = (19583/254016 + (24401/193536) * nu + (31/288) * nu**2) * theta_m1_2
        PN_2_5 = (-(11891/53760) + (109/1920) * nu) * np.pi * theta_m5_8
        PN_3 = (
            -10052469856691/6008596070400
            + (1/6) * np.pi**2
            + (107/420) * C
            - (107/3360) * np.log(theta/256)
            + (3147553127/780337152 - (451/3072) * np.pi**2) * nu
            - (15211/442368) * nu**2
            + (25565/331776) * nu**3
        ) * theta_m3_4

    elif _is(order, 3.5):
        PN_1 = (743/4042 + (11/48) * nu) * theta_m1_4
        PN_1_5 = -(1/5) * np.pi * theta_m3_8
        PN_2 = (19583/254016 + (24401/193536) * nu + (31/288) * nu**2) * theta_m1_2
        PN_2_5 = (-(11891/53760) + (109/1920) * nu) * np.pi * theta_m5_8
        PN_3 = (
            -10052469856691/6008596070400
            + (1/6) * np.pi**2
            + (107/420) * C
            - (107/3360) * np.log(theta/256)
            + (3147553127/780337152 - (451/3072) * np.pi**2) * nu
            - (15211/442368) * nu**2
            + (25565/331776) * nu**3
        ) * theta_m3_4
        PN_3_5 = (
            -(113868647/433520640)
            - (31821/143360) * nu
            + (294941/3870720) * nu**2
        ) * np.pi * theta_m7_8

    else:
        raise ValueError("order must be one of {1, 1.5, 2, 2.5, 3, 3.5}")

    # x = 1/4 * theta^(-1/4) * (sum PN terms)
    x = 0.25 * theta_m1_4 * (PN_0 + PN_1 + PN_1_5 + PN_2 + PN_2_5 + PN_3 + PN_3_5)

    omegaS = (c**3) / (G * m * Msun) * (x ** (3.0 / 2.0))
    omega_gw = 2.0 * omegaS
    fgw = omega_gw / (2.0 * np.pi)

    # 0PN (leading order)
    x_0pn = 0.25 * theta_m1_4
    omegaS_0pn = (c**3) / (G * m * Msun) * (x_0pn ** (3.0 / 2.0))
    omega_gw_0pn = 2.0 * omegaS_0pn
    fgw_0pn = omega_gw_0pn / (2.0 * np.pi)

    df = np.abs(fgw - fgw_0pn)
    return fgw, fgw_0pn, df



