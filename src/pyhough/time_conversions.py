import numpy as np

def gps2mjd(tgps):
    """
    Convert GPS time (seconds) to Modified Julian Date (days).

    Parameters
    ----------
    tgps : float or array-like
        GPS time in seconds.

    Returns
    -------
    mjd : ndarray
        Modified Julian Date (days).
    """

    tgps = np.asarray(tgps, dtype=float)

    t0 = 44244.0  # MJD at GPS epoch (6-Jan-1980 00:00:00)

    mjd = tgps / 86400.0 + t0

    # Leap second correction (GPS linked to TAI, offset from UTC)
    mjd = mjd - (leap_seconds(mjd) - 19.0) / 86400.0

    return mjd


# MJD effective dates when TAI-UTC stepped by +1s (from your MATLAB table)
_LEAP_MJD = np.array([
    41317, 41499, 41683, 42048, 42413, 42778, 43144, 43509, 43874,
    44786, 45151, 45516, 46247, 47161, 47892, 48257, 48804, 49169,
    49534, 50083, 50630, 51179, 53736, 54832, 56109, 57204, 57754
], dtype=float)

# nls at/after the last entry in the table above (2017-01-01): TAI-UTC = 37 s
# If new leap seconds occur, you must update both _LEAP_MJD and this value.
_LEAP_MAX = 37


def leap_seconds(mjd):
    """
    Number of leap seconds (TAI-UTC in seconds) applicable at a given MJD.

    Parameters
    ----------
    mjd : float or array-like
        Modified Julian Date (days).

    Returns
    -------
    nls : float or ndarray
        TAI-UTC (seconds). Same shape as input.
    """
    mjd = np.asarray(mjd, dtype=float)

    # Count how many leap dates are strictly less than mjd
    # (MATLAB code uses: if mjd > leaptimes(i) then break)
    n_before = np.searchsorted(_LEAP_MJD, mjd, side="right")

    # At mjd beyond the last leap date, n_before == len(_LEAP_MJD) -> returns _LEAP_MAX
    # At earlier mjd, subtract how many steps haven't happened yet.
    nls = _LEAP_MAX - (len(_LEAP_MJD) - n_before)

    # Return scalar if scalar input
    if nls.shape == ():
        return float(nls)
    return nls


def tdt2tdb(mjd):
# % TDT2TDB  seconds to add to tdt (terrestrial dymamical time (TAI corrected)) 
# %          to have tdb (barycentric dynamical time)
# %
# %   mjd   mjd value (days)
# %
# %   tdb   seconds to add to the tdt

    JD = mjd + 2400000.5
    g = np.mod(357.53 + 0.98560028 * (JD - 2451545.0),360) * np.pi/180
    tdb = 0.001658 * np.sin(g) + 0.000014 * np.sin(2*g)
    return tdb

def gmst(t):
    """
    Greenwich Mean Sidereal Time (GMST) in hours.

    Parameters
    ----------
    t : float or ndarray
        Time in JD or MJD.
        If t < 1e6, it is assumed to be MJD and converted to JD.

    Returns
    -------
    st : float or ndarray
        GMST in hours (range [0, 24)).
    """

    t = np.asarray(t, dtype=float)

    # Convert MJD -> JD if needed
    jd = np.where(t < 1_000_000, t + 2400000.5, t)

    jd0 = np.floor(jd - 0.5) + 0.5
    h = (jd - jd0) * 24.0

    d  = jd  - 2451545.0
    d0 = jd0 - 2451545.0
    T  = d / 36525.0

    st = (
        6.697374558
        + 0.06570982441908 * d0
        + 1.00273790935 * h
        + 0.000026 * T**2
    )

    return np.mod(st, 24.0)

print(gmst(6.008867471064815e+04))