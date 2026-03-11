import numpy as np
import pyhough
import matplotlib.pyplot as plt
from typing import Tuple


def hfdf_hough_transients(peaks, hm_job):
    Day_inSeconds = 86400

    gridk = np.squeeze(hm_job['gridk'])
    minf0 = hm_job['minf']
    maxf0 = hm_job['maxf']
    df = hm_job['df']
    enh = hm_job['frenh']
    braking_index = hm_job['n']
    epoch = hm_job['epoch']
    poww = braking_index - 1

    if braking_index == 5 or braking_index == 3 or braking_index == 7:
        pass
    else:
        # print('chirp, flipping spindowns to spinups')
        gridk = -gridk

    peaks = peaks.copy()
    peaks[0, :] = Day_inSeconds * (peaks[0, :] - epoch)

    if braking_index == 1:
        x = np.log(peaks[1, :])
        minx = np.min(x)
        maxx = np.max(x)
        dx = df * 1 / maxf0
        poww = 1 ### not physical, just to make codes work
    else:
        x = peaks[1, :] ** -poww
        minx = 1 / maxf0 ** poww
        maxx = 1 / minf0 ** poww

        if maxx < minx:
            minx, maxx = maxx, minx

        dx = (poww) * df * 1 / (maxf0) ** (braking_index)
        dx = abs(dx)

    ddx = dx / enh
    inix, finx = minx, maxx

    nbin_x = int(np.ceil((abs(finx - inix)) / ddx))

    ii = np.squeeze(np.argwhere(np.diff(peaks[0, :])))
    nTimeSteps = len(ii)

    ii0 = 0

    binh_df0 = np.zeros((len(gridk), nbin_x))

    for it in range(nTimeSteps):
        x0_a = ((x[ii0:ii[it]+1] - inix) / ddx)
        t = peaks[0, ii0]
        tddx = t / ddx
        
        for id in range(len(gridk)):
            td = gridk[id] * tddx * poww
            inds = np.round(x0_a - td).astype(int)
            ind_of_inds = np.argwhere(inds >= 0)
            a = inds[ind_of_inds]
            log_inds = a <= nbin_x-1
            a = a[log_inds]
            binh_df0[id, a] = binh_df0[id,a] + 1

        ii0 = ii[it] + 1

    hm_job = hm_job.copy() 
    hm_job['dx'] = dx
    hm_job['gridx'] = np.arange(inix, finx, ddx)

    return binh_df0, hm_job


def LongT_GENERALIZED_fasthough(
    peaks: np.ndarray,
    hm_job: dict,
) -> Tuple[np.ndarray, dict]:
    """
    Python translation of:

        [hfdf_map, hm_job] = LongT_GENERALIZED_fasthough(peaks, hm_job)

    Returns
    -------
    hfdf_map : np.ndarray
        2D Hough map array of shape (nbin_fbandx, len(gridk))
    hm_job : dict
        Same object, updated in-place with:
            - 'dx'
            - 'which_hough'
    """

    peaks = np.asarray(peaks)
    if peaks.shape[0] != 3:
        raise ValueError("peaks must have shape (3, N)")

    n = float(hm_job["n"])
    pow_ = n - 1.0
    epoch = hm_job["epoch"]

    n_peaks = peaks.shape[1]

    # Weights
    weights = np.ones(n_peaks)
    weights[peaks[1, :] < 0] = 0.0

    # Convert time to seconds relative to epoch
    t_sec = (peaks[0, :] - epoch) * 86400.0
    Tmax = np.max(np.abs(t_sec)) if n_peaks > 0 else 0.0

    minf0 = hm_job["minf"]
    maxf0 = hm_job["maxf"]
    df = hm_job["df"]
    enh = hm_job.get("frenh", 1.0)
    fr = peaks[1, :].astype(float)

    # ----- x transform -----
    if n == 1.0:
        xx = np.log(fr)
        dx = df / maxf0
        ddx = dx / enh
        minx0 = np.floor(np.log(minf0) / ddx) * ddx
        maxx0 = np.ceil(np.log(maxf0) / ddx) * ddx
        pow_ = 1.0
    else:
        xx = fr ** (-pow_)
        dx = pow_ * df / (maxf0 ** n)
        ddx = dx / enh
        minx0 = 1.0 / (maxf0 ** pow_)
        maxx0 = 1.0 / (minf0 ** pow_)
        if maxx0 < minx0:
            minx0, maxx0 = maxx0, minx0
        dx = abs(dx)

    gridk = np.asarray(hm_job["gridk"], dtype=float)

    if n == 11.0 / 3.0:
        gridk = -gridk

    sd_belt = (
        int(np.ceil(Tmax * pow_ * np.max(np.abs(gridk)) / ddx) + 10)
        if gridk.size
        else 10
    )

    nbin_fbandx = int(np.ceil(abs(maxx0 - minx0) / ddx))
    inix = minx0 - sd_belt * ddx
    nbin_x = nbin_fbandx + 2 * sd_belt

    # ----- Normalize k grid -----
    if len(gridk) >= 2:
        dk = np.min(np.abs(np.diff(gridk)))
        if dk == 0:
            dk = 1.0
    else:
        dk = 1.0

    xx_norm = (xx - inix) / ddx
    tt_norm = t_sec * dk * pow_ / ddx
    slopes = gridk / dk

    # ----- Build Hough map -----
    hfdf_cols = []

    for s in slopes:
        idx = np.round(xx_norm - tt_norm * s).astype(int)

        valid = (idx >= 0) & (idx < nbin_x) & (weights != 0)
        col = np.bincount(idx[valid], weights=weights[valid], minlength=nbin_x)

        # Remove spindown belts
        col = col[sd_belt: sd_belt + nbin_fbandx]
        hfdf_cols.append(col)

    if hfdf_cols:
        hmap = np.vstack(hfdf_cols)
    else:
        hmap = np.zeros((0, nbin_fbandx))

    # ---- Update hm_job in place ----
    hm_job['gridx'] = np.arange(minx0,maxx0,dx)
    hm_job["dx"] = dx
    hm_job["which_hough"] = "gfh"

    return hmap, hm_job

def make_hm_job_struct(minf, maxf, TFFT, dur, n, ref_perc_time, gridk, epoch):
    
    hm_job = {
        'minf': minf,               # minimum frequency to do the Hough on
        'maxf': maxf,               # maximum frequency to do the Hough on
        'df': 1 / TFFT,             # step in frequency
        'dur': dur,
        'patch': [0, 0],
        'n': n,
        'ref_perc_time': ref_perc_time,
        'frenh': 1,
        'gridk': gridk,
        'epoch': epoch
    }

    return hm_job


def andrew_long_transient_grid_k(Tfft, f0range, f0dotrange, tobs, nb):
    f0min = min(f0range)
    f0max = max(f0range)
    f0dot_min = f0dotrange[0]
    f0dot_max = f0dotrange[1]
    log10fdotmin = np.log10(abs(f0dotrange[0]))
    log10fdotmax = np.log10(abs(f0dotrange[1]))

    randf = f0min + (f0max - f0min) * np.random.rand(10000)
    log10randfdot = log10fdotmin + (log10fdotmax - log10fdotmin) * np.random.rand(10000)

    each_k_step = []
    nk = []
    gridk = []

    for i in range(1):
        kmin = f0dot_min / f0max**nb
        kmax = f0dot_max / f0min**nb

        newk = kmax
        j = 1
        k = []

        while newk >= kmin:
            dk = newk * ((1 + 1 / (Tfft * f0max))**(-nb) - 1)
            each_k_step.append(dk)

            if j == 1:
                k.append(newk)
            else:
                k.append(newk + dk)

            newk = k[j - 1]
            j += 1

        nk.append(len(k))
        gridk.append(k)

    gridk = np.flip(gridk)
    each_k_step = np.abs(np.flip(each_k_step))

    return gridk, each_k_step

def cbc_shorten_gridk(gridk, mink, maxk, frac_around=0.15):
    factor = 1 + frac_around
    kmin = mink / factor
    kmax = maxk * factor
    inddd = np.argmin(np.abs(kmin - gridk))
    ind2 = np.argmin(np.abs(kmax - gridk))
    reduced_gridk = gridk[inddd:ind2 + 1]
    return reduced_gridk

def get_f0_from_x0(x0, n):
    f0 = x0**(-1 / (n - 1))
    return f0

import numpy as np
import numba as nb


def LongT_GENERALIZED_fasthough_nonuni(peakss, hm_job):
    """
    Creates a x/k Hough map with non-uniform x-grid binning
    
    This code takes as input a peakmap and first transforms t/f --> t/x
    according to the braking index, then maps t/x --> x/k using the Hough
    
    Parameters:
    -----------
    peakss : ndarray
        peaks(3,n) - peaks of the peakmap as [t,fr,amp] 
        (fr corrected for the Doppler effect)
        Row 0: time (MJD)
        Row 1: frequency (Hz)
        Row 2: amplitude (not used)
    
    hm_job : dict
        Hough map structure containing:
            'minf' : minimum frequency of Hough map (Hz)
            'maxf' : maximum frequency of Hough map (Hz)
            'df' : frequency resolution (1/TFFT) (Hz)
            'dur' : duration of peakmap (s)
            'patch' : [Longitude Latitude] (ecliptic)
            'n' : braking index
            'ref_perc_time' : percentile of reference time [0,1]
            'frenh' : frequency enhancement (1)
            'gridk' : grid on constant k parameter
            'epoch' : reference time (MJD)
    
    Returns:
    --------
    hfdf : ndarray
        Hough map (transposed histogram)
    hm_job : dict
        Updated hough map structure with additional fields:
            'gridx' : x-grid values
            'dx' : spacing in x grid (Hz^{1-n})
            'which_hough' : 'gfh_nonuni'
    """
    
    Day_inSeconds = 86400
    
    gridk = hm_job['gridk'].copy()
    braking_index = hm_job['n']
    
    # Flip spindowns to spinups for certain braking indices
    if braking_index not in [5, 3, 7]:
        # disp('chirp, flipping spindowns to spinups')
        gridk = -gridk
    
    pow_val = braking_index - 1
    
    n2 = peakss.shape[1]
    weights = np.ones(n2)
    
    epoch = hm_job['epoch']
    tpeaks = Day_inSeconds * (peakss[0, :] - epoch)
    
    minf0 = hm_job['minf']
    maxf0 = hm_job['maxf']
    df = hm_job['df']
    enh = hm_job['frenh']
    
    if braking_index == 1:  # case of pulsar winds
        xpeaks = np.log(peakss[1, :])
        pow_val = 1  # not physical, negates pow in each expression
    else:
        xpeaks = peakss[1, :] ** (-pow_val)
    
    # Create non-uniform grid
    freq_grid = np.arange(minf0, maxf0 + df, df)
    gridx = np.flip(1.0 / (freq_grid ** pow_val))
    
    # Call the fast vectorized version
    binh_df0 = original_version_nonuni_fast(
        xpeaks, tpeaks, gridk, gridx, weights, braking_index
    )
    
    hfdf = binh_df0
    
    # Update hm_job with output parameters
    hm_job['gridx'] = gridx[:-1]
    hm_job['dx'] = np.diff(gridx)
    hm_job['which_hough'] = 'gfh_nonuni'
    
    return hfdf, hm_job


def original_version_nonuni_fast(xpeaks, tpeaks, gridk, gridx, weights, braking_index):
    """
    Fully vectorized fast non-uniform x-grid binning using list comprehension
    Inspired by LongT_GENERALIZED_fasthough vectorization
    
    Parameters:
    -----------
    xpeaks : array_like
        Transformed peak frequencies (x = f^(-pow))
    tpeaks : array_like
        Peak times in seconds (relative to epoch)
    gridk : array_like
        Grid of k parameter values to search
    gridx : array_like
        Non-uniform x-grid edges
    weights : array_like
        Weight for each peak (usually all 1s)
    braking_index : float
        Braking index
    
    Returns:
    --------
    binh_df0_orig : ndarray
        Hough map of shape (nbin_k, num_bins)
    """
    
    pow_val = braking_index - 1
    num_bins = len(gridx) - 1
    
    # Create bin edges from the grid points
    bin_edges = np.concatenate([gridx, [np.inf]])
    
    # Normalize time by smallest k step for numerical stability
    dk_diff = np.diff(gridk)
    if len(dk_diff) > 0:
        dk = np.min(np.abs(dk_diff))
    else:
        dk = 1.0
    
    slopes = gridk / dk
    tt_norm = tpeaks * dk * pow_val
    
    # THE KEY VECTORIZATION: Use list comprehension to process all k values at once
    # This is the same trick as LongT_GENERALIZED_fasthough
    hfdf_list = [
        discretize_and_accumulate(xpeaks, tt_norm, slope, bin_edges, num_bins, weights)
        for slope in slopes
    ]
    
    # Convert list to matrix (stack as columns, then transpose)
    binh_df0_orig = np.column_stack(hfdf_list).T
    
    return binh_df0_orig


def discretize_and_accumulate(xx, tt_norm, slope, bin_edges, num_bins, weights):
    """
    Compute x0 values for this slope and bin them
    
    Parameters:
    -----------
    xx : array_like
        X values (transformed frequencies)
    tt_norm : array_like
        Normalized time values
    slope : float
        Current k slope value
    bin_edges : array_like
        Bin edges for discretization
    num_bins : int
        Number of bins
    weights : array_like
        Weight for each peak
    
    Returns:
    --------
    counts : ndarray
        Histogram counts for this slope
    """
    # Compute x0 values for this slope
    x0s = xx - tt_norm * slope
    
    # Discretize into bins
    # np.digitize returns 1-based indices, subtract 1 for 0-based
    bin_idx = np.digitize(x0s, bin_edges) - 1
    
    # Filter valid bins
    valid = (bin_idx >= 0) & (bin_idx < num_bins)
    
    # Accumulate with weights
    if np.any(valid):
        counts = np.bincount(
            bin_idx[valid], 
            weights=weights[valid], 
            minlength=num_bins
        )[:num_bins]
    else:
        counts = np.zeros(num_bins)
    
    return counts

def plot_hm(hmap,info,physical=True,lab='number count'):
    if physical:
        mcsss = pyhough.physics.calc_mc_with_k(info['gridk'])
        fffss = pyhough.gfh.get_f0_from_x0(info['gridx'],info['n'])
    else:
        mcsss = info['gridk']
        fffss = info['gridx']
    fig, ax = plt.subplots()#figsize=(0.8 * 16, 0.8 * 9))
    if physical:
        ax.set(ylabel=r"$\mathcal{M}$ $[M_\odot]$", xlabel=r"frequency [Hz]")
    else:
        xlab,ylab = get_hough_axis_labels(info['n'])
        ax.set(ylabel=ylab, xlabel=xlab)
    c = ax.pcolormesh(
        fffss,
        np.squeeze(mcsss),
        hmap,
        cmap="inferno",
        shading="nearest",
    )
    fig.colorbar(c, label=lab)
    plt.yscale('log')
    plt.tight_layout()

from fractions import Fraction

def format_power(p):
    """
    Format exponent p for LaTeX, handling integers and fractions cleanly.
    """
    frac = Fraction(p).limit_denominator()

    if frac.denominator == 1:
        return f"{frac.numerator}"
    else:
        return r"\frac{" + f"{frac.numerator}" + "}{" + f"{frac.denominator}" + "}"

def get_hough_axis_labels(n):
    """
    Returns LaTeX strings for x0 and k axis labels with correct units.
    """

    # exponents
    x_exp = -(n - 1)
    k_exp = 2 - n

    x_exp_str = format_power(x_exp)
    k_exp_str = format_power(k_exp)

    xlabel = rf"$x_0\;[\mathrm{{Hz}}^{{{x_exp_str}}}]$"
    ylabel = rf"$\mathcal{{k}}\;[\mathrm{{Hz}}^{{{k_exp_str}}}]$"

    return xlabel, ylabel

def vary_p0_hfdf_compute_mu_sigma_nonuni_grids(
    gridx,
    gridk,
    n,
    TFFT,
    fmin,
    fmax,
    tim,
    epoch,
    p0_sft,
):
    """
    Compute MU(x,k) and ST(x,k)=sqrt(VAR) for GFH on nonuniform x/k grids,
    with per-SFT peak probability p0_sft (no weights argument needed).

    Parameters
    ----------
    gridx : array_like, shape (Nx,) or (Nx,1)
        Nonuniform x grid centers (must be strictly increasing).
    gridk : array_like, shape (Nk,) or (Nk,1)
        k grid values.
    n : float
        Braking index; x = f^(-(n-1)).
    TFFT : float
        Coherence time in seconds; df = 1/TFFT.
    fmin, fmax : float
        Frequency band in Hz. (Uses MATLAB-style inclusive fmin:df:fmax behavior.)
    tim : array_like, shape (Nt,)
        SFT times in days (e.g. MJD).
    epoch : float
        Reference time in days (MJD).
    p0_sft : array_like, shape (Nt,) or scalar
        Per-SFT peak probability. If scalar, it is broadcast to Nt.

    Returns
    -------
    MU : ndarray, shape (Nx, Nk)
        Expected mean Hough counts per (x,k).
    ST : ndarray, shape (Nx, Nk)
        Standard deviation per (x,k), computed as sqrt(sum p0*(1-p0)).
    """
    # unpack / basics
    tsec = 86400.0 * (np.asarray(tim).reshape(-1) - float(epoch))  # (Nt,)
    df = 1.0 / float(TFFT)
    pow_ = float(n) - 1.0

    # shape checks
    gridx = np.asarray(gridx).reshape(-1)
    gridk = np.asarray(gridk).reshape(-1)
    p0_sft = np.asarray(p0_sft).reshape(-1)

    Nx = gridx.size
    Nk = gridk.size
    Nt = tsec.size

    if p0_sft.size == 1:
        p0_sft = np.full(Nt, float(p0_sft[0]))
    if p0_sft.size != Nt:
        raise ValueError("p0_sft must be Nt×1 (per SFT) or a scalar.")
    if Nx < 2:
        raise ValueError("gridx must have at least 2 elements.")

    # x-bin edges (Nx+1), centered about gridx (works for nonuniform grid)
    xedges = np.empty(Nx + 1, dtype=float)
    xedges[0] = gridx[0] - 0.5 * (gridx[1] - gridx[0])
    xedges[1:-1] = 0.5 * (gridx[:-1] + gridx[1:])
    xedges[-1] = gridx[-1] + 0.5 * (gridx[-1] - gridx[-2])

    if not np.all(np.diff(xedges) > 0):
        raise ValueError("xedges must be strictly increasing.")

    # outputs
    MU = np.zeros((Nx, Nk), dtype=float)
    VAR = np.zeros((Nx, Nk), dtype=float)

    # frequency grid (match Hough) -- MATLAB fmin:df:fmax inclusive
    # Avoid float-step drift by computing count explicitly.
    Nf = int(np.floor((fmax - fmin) / df + 1.0 + 1e-12))
    f_sample = fmin + df * np.arange(Nf, dtype=float)  # (Nf,)
    x_sample = f_sample ** (-pow_)  # (Nf,)

    # expand per-SFT p0 across all freq bins → (Nt*Nf,)
    p0_vec = np.repeat(p0_sft, Nf)
    q0_vec = p0_vec * (1.0 - p0_vec)

    # helper: discretize-like binning (MATLAB convention: left-open, right-closed),
    # returning 1..Nx, or 0 for out-of-range (we'll treat 0 as invalid).
    # Using np.searchsorted with side="left" implements right-closed behavior:
    # idx = searchsorted(edges, x, 'left') gives i such that edges[i-1] < x <= edges[i]
    def _discretize_right_closed(x, edges):
        idx = np.searchsorted(edges, x, side="left")  # 0..Nx+1
        # valid bins are 1..Nx
        valid = (idx >= 1) & (idx <= Nx)
        return idx, valid

    # main loop over k (vectorized over t and f)
    for ik, k in enumerate(gridk):
        # map (t,f) → x0: x0 = f^(-pow) - pow*k*t
        # shape: (Nt, Nf)
        x0_grid = x_sample[None, :] - pow_ * float(k) * tsec[:, None]
        x0_vec = x0_grid.ravel()

        bin_idx, valid = _discretize_right_closed(x0_vec, xedges)

        # accumarray equivalent with bincount; convert 1..Nx -> 0..Nx-1
        b = bin_idx[valid] - 1
        MU[:, ik] = np.bincount(b, weights=p0_vec[valid], minlength=Nx)
        VAR[:, ik] = np.bincount(b, weights=q0_vec[valid], minlength=Nx)

    ST = np.sqrt(VAR)
    return MU, ST




