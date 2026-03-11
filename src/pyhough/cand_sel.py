import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


def robstat(x,p=0.01):
    """
    Python translation of MATLAB robstat(x,p).

    Parameters
    ----------
    x : array-like
        Data.
    p : float or array-like
        Probability/probabilities in (0,1). Example: 0.01.

    Returns
    -------
    m : np.ndarray
        [median, sigma_rob, pct(p1), pct(1-p1), pct(p2), pct(1-p2), ...]
        where sigma_rob = median(|x-median|)/0.6745 (normal-equivalent 1-sigma).
    """
    x = np.asarray(x, dtype=float).ravel()
    p_arr = np.atleast_1d(p).astype(float)

    med = np.median(x)
    sig = np.median(np.abs(x - med)) / 0.6745

    out = [med, sig]
    for pi in p_arr:
        out.append(np.percentile(x, 100.0 * pi))
        out.append(np.percentile(x, 100.0 * (1.0 - pi)))
    return np.asarray(out, dtype=float)

def select_cands_transients_nonuni(
# def hfdf_transients_peak_selection_like_FH_for_nonuni(
    hfdf: np.ndarray,
    hm_job: Dict,
    kcand: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Python translation of hfdf_transients_peak_selection_like_FH_for_nonuni,
    adapted to:
      - hfdf is a plain matrix (no gd2), shape (len(gridk), len(gridx)) OR (len(gridx), len(gridk))
      - dn removed (not needed)

    Assumptions / conventions
    -------------------------
    The MATLAB code operates on y = y_gd2(g) where y is (Nx, Nk) and then uses max(y')
      -> for each x-bin, take max over k.

    Here we enforce:
      y.shape == (Nx, Nk)  (x along axis 0, k along axis 1)

    If your hfdf is stored as (Nk, Nx), it will be transposed automatically if it matches grids.

    Required hm_job keys
    --------------------
    hm_job['n']     : braking index
    hm_job['frenh'] : over-resolution factor
    hm_job['minf'], hm_job['maxf']
    hm_job['gridk'] : (Nk,) array
    hm_job['dur']   : duration
    hm_job['epoch'] : reference time
    hm_job['df']    : 1/TFFT (or your df)
    hm_job['gridx'] : (Nx,) array of x-bin centers/values corresponding to hfdf x-axis

    Optional hm_job keys
    --------------------
    hm_job['patch'] : (2,) [lon, lat]
    hm_job['MU']    : (Nx, Nk) analytic mean map
    hm_job['ST']    : (Nx, Nk) analytic std map

    Returns
    -------
    cand : np.ndarray
        Shape (17, kcand). Columns are candidates (unused columns stay 0).
        Fields follow your MATLAB with added MU/ST slots at rows 16-17.
    job_info : dict
        Contains robstat info and other bookkeeping.
    """
    # ---- pull params ----
    braking_index = float(hm_job["n"])
    pow_ = braking_index - 1.0

    mno = float(hm_job["frenh"]) * 2.0
    mode = 2

    frini = float(hm_job["minf"])
    frfin = float(hm_job["maxf"])

    # x-range (used in MATLAB cut_gd2); here used to crop in x if desired
    xmin = 1.0 / (frfin ** pow_)
    xmax = 1.0 / (frini ** pow_)

    gridk = np.asarray(hm_job["gridk"], dtype=float)
    dur = hm_job["dur"]
    num_kbins = gridk.size

    # dx in your MATLAB is hm_job.dx, but you later index dx(iii) as if it's per-x-bin.
    # In many of your Python snippets dx is scalar; in your MATLAB snippet it behaves like vector.
    # We'll accept either.
    dx = hm_job.get("dx", np.nan)

    gridx = np.asarray(hm_job["gridx"], dtype=float)

    patch = np.asarray(hm_job.get("patch", [0.0, 0.0]), dtype=float)
    tini = hm_job["epoch"]

    MU = hm_job.get("MU", None)
    ST = hm_job.get("ST", None)
    use_analytic = (MU is not None) and (ST is not None)

    # ---- ensure hfdf orientation matches (Nx, Nk) ----
    hfdf = np.asarray(hfdf)
    Nx = gridx.size
    Nk = gridk.size

    if hfdf.shape == (Nx, Nk):
        y_full = hfdf
    elif hfdf.shape == (Nk, Nx):
        y_full = hfdf.T
    else:
        raise ValueError(
            f"hfdf shape {hfdf.shape} does not match (Nx,Nk)=({Nx},{Nk}) or (Nk,Nx)=({Nk},{Nx})"
        )

    # ---- crop in x like cut_gd2(hfdf,[xmin,xmax],...) ----
    # MATLAB uses cut_gd2 to restrict x-range; we do the same via boolean mask on gridx.
    xmask = (gridx >= xmin) & (gridx <= xmax)
    if not np.any(xmask):
        # fall back: do not crop, but make it explicit
        xmask = np.ones_like(gridx, dtype=bool)

    y = y_full[xmask, :]
    fr = gridx[xmask]  # "fr" in MATLAB variable naming is x-grid here

    # also crop MU/ST if provided
    if use_analytic:
        MUc = np.asarray(MU)
        STc = np.asarray(ST)
        if MUc.shape != y_full.shape or STc.shape != y_full.shape:
            raise ValueError("hm_job['MU'] and ['ST'] must have same shape as hfdf in (Nx,Nk) convention.")
        MUc = MUc[xmask, :]
        STc = STc[xmask, :]
    else:
        MUc = STc = None

    # ---- sd_steps for nonuniform k grid ----
    sd_steps = np.diff(gridk)
    if sd_steps.size == 0:
        sd_steps = np.array([0.0])
    else:
        sd_steps = np.concatenate([sd_steps, [sd_steps[-1]]])

    # ---- per-x maxima over k ----
    # MATLAB: [ym,im]=max(y');  -> ym,im length Nx, where im is argmax over k
    # Here y is (Nx,Nk)
    im = np.argmax(y, axis=1)          # (Nx,)
    ym = y[np.arange(y.shape[0]), im]  # (Nx,)

    N = ym.size
    df = N / kcand
    # MATLAB: ix=round(1:df:N); ix=[ix N+1];
    # In Python 0-based, we build boundaries in [0..N]
    ix = np.rint(np.arange(0, N, df)).astype(int)
    if ix.size == 0 or ix[0] != 0:
        ix = np.concatenate([[0], ix])
    if ix[-1] != N:
        ix = np.concatenate([ix, [N]])

    # robust stats
    robst_all = robstat(y.ravel(), 0.01)
    robmedtot = robst_all[0]

    robmed = np.zeros(kcand, dtype=float)
    robstd = np.zeros(kcand, dtype=float)

    # compute local robust med/std in widened windows like MATLAB
    # MATLAB uses ix(i-1) : ix(i+2)-1 for i=2..kcand-1 (1-based).
    # We'll emulate on blocks in 0-based. We define block i as [ix[i], ix[i+1]).
    for i in range(kcand):
        # window spans blocks i-1 to i+1 inclusive (i-1 .. i+1)
        lo_block = max(i - 1, 0)
        hi_block = min(i + 2, len(ix) - 1)  # exclusive boundary index in ix
        lo = ix[lo_block]
        hi = ix[hi_block]
        seg = ym[lo:hi]
        r = robstat(seg, 0.01)
        robmed[i] = r[0]
        robstd[i] = r[1] if r[1] > 0 else np.nan

    job_info: Dict = {}
    job_info["robst"] = robst_all
    job_info["robmed"] = robmed
    job_info["robstd"] = robstd

    # ---- candidate array ----
    # MATLAB alloc cand=zeros(15,kcand) but then fills up to 17.
    cand = np.zeros((17, kcand), dtype=float)

    jj = 0
    for i in range(kcand):
        if robmed[i] <= 0:
            continue

        lo = ix[i]
        hi = ix[i + 1]
        if hi <= lo:
            continue

        yy = ym[lo:hi].copy()
        ma = float(np.max(yy))
        ima = int(np.argmax(yy))  # local index
        if not (ma > robmed[i] and ma > robmedtot / 2.0):
            continue

        # global x index in cropped arrays
        iii = lo + ima
        k_idx = int(im[iii])

        # emit primary candidate
        if jj < kcand:
            cand[0, jj] = fr[iii]          # x
            cand[1, jj] = patch[0]         # lon
            cand[2, jj] = patch[1]         # lat
            cand[3, jj] = (gridk[k_idx])     # k
            cand[4, jj] = ma               # amp (number count / CR placeholder)
            cand[5, jj] = ma               # CR (as in your snippet)
            cand[6, jj] = braking_index
            cand[7, jj] = sd_steps[k_idx]
            cand[8, jj] = tini
            cand[9, jj] = dur
            cand[10, jj] = hm_job["df"]
            cand[11, jj] = 0.0             # dn removed -> set 0
            # dx(iii): allow scalar or vector
            cand[12, jj] = dx[iii] if hasattr(dx, "__len__") else float(dx)
            cand[13, jj] = cand[0, jj] ** (-1.0 / (cand[6, jj] - 1.0))  # recovered f0
            cand[14, jj] = cand[3, jj] * (cand[13, jj] ** cand[6, jj])  # recovered fdot0
            if use_analytic:
                cand[15, jj] = MUc[iii, k_idx]
                cand[16, jj] = STc[iii, k_idx]
            jj += 1

        # optional second candidate sufficiently far in k within same x-block
        if mode == 2 and jj < kcand:
            i1 = max(ima - int(mno), 0)
            i2 = min(ima + int(mno) + 1, yy.size)
            yy[i1:i2] = 0.0
            ma1 = float(np.max(yy))
            ima1 = int(np.argmax(yy))

            if abs(ima1 - ima) > 2 * mno and ma1 > robmed[i]:
                iii2 = lo + ima1
                k_idx2 = int(im[iii2])

                cand[0, jj] = fr[iii2]
                cand[1, jj] = patch[0]
                cand[2, jj] = patch[1]
                cand[3, jj] = (gridk[k_idx2])
                cand[4, jj] = ma1
                cand[5, jj] = ma1
                cand[6, jj] = braking_index
                cand[7, jj] = sd_steps[k_idx2]
                cand[8, jj] = tini
                cand[9, jj] = dur
                cand[10, jj] = hm_job["df"]
                cand[11, jj] = 0.0
                cand[12, jj] = dx[iii2] if hasattr(dx, "__len__") else float(dx)
                cand[13, jj] = cand[0, jj] ** (-1.0 / (cand[6, jj] - 1.0))
                cand[14, jj] = cand[3, jj] * (cand[13, jj] ** cand[6, jj])
                if use_analytic:
                    cand[15, jj] = MUc[iii2, k_idx2]
                    cand[16, jj] = STc[iii2, k_idx2]
                jj += 1

    # MATLAB post-flip for chirp
    # if abs(braking_index - 11.0 / 3.0) < 1e-12:
    #     cand[3, :] *= -1.0

    # ---- job_info summary ----
    job_info["ncand"] = jj
    job_info["dx"] = dx
    job_info["num_kbins"] = num_kbins
    job_info["n"] = braking_index
    job_info["df"] = hm_job["df"]
    job_info["fap"] = cand.shape[1] / (y.shape[0] * y.shape[1])

    return cand, job_info



def coin_inj_cand(
    cand2: np.ndarray,
    theo_x0: float,
    theo_kn: float,
    coin_dist: float = 3.0,
) -> Tuple[bool, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Coincidence test between injection parameters and candidates.

    Parameters
    ----------
    cand2 : ndarray
        Candidate matrix (shape: npar x Ncand)
        Expected:
            cand2[0,:] = x0
            cand2[3,:] = k
            cand2[5,:] = CR (used for tie-breaking)
            cand2[7,:] = dk
            cand2[12,:] = dx
    theo_x0 : float
        Injected/theoretical x0
    theo_kn : float
        Injected/theoretical k
    coin_dist : float, optional
        Coincidence threshold (default = 3)

    Returns
    -------
    found : bool
        True if any candidate within coin_dist
    best_cand : ndarray
        Best matching candidate (column vector)
    mindist : float
        Distance of best candidate
    dist : ndarray
        Distance of all candidates
    dist_each_parm_best_inj : ndarray
        Contribution from each parameter (x0, k)
    """

    # Candidate parameters
    cand_x0 = cand2[0, :]
    cand_kn = cand2[3, :]
    cand_dk = cand2[7, :]
    cand_dx = cand2[12, :]

    # Differences
    diff_x0 = np.abs(cand_x0 - theo_x0)
    diff_kn = np.abs(cand_kn - theo_kn)

    # Normalized distance terms
    term1 = (diff_x0 / cand_dx) ** 2
    term2 = (diff_kn / cand_dk) ** 2

    dist = np.sqrt(term1 + term2)

    inds_close = np.where(dist <= coin_dist)[0]

    if len(inds_close) == 0 or len(inds_close) == 1:
        ind = np.argmin(dist)
        mindist = dist[ind]
    else:
        # choose highest CR among close candidates
        ind_best = np.argmax(cand2[5, inds_close])
        ind = inds_close[ind_best]
        mindist = dist[ind]

    dist_each_parm_best_inj = np.sqrt(
        np.array([term1[ind], term2[ind]])
    )

    best_cand = cand2[:, ind]

    found = np.any(dist <= coin_dist)

    return found, best_cand, mindist, dist, dist_each_parm_best_inj



def plot_CR_histogram_with_inset(
    CR,
    inj_CR,
    bins=100,
    ax=None
):
    """
    Plot histogram of CR values with inset zoom around injected CR.

    Parameters
    ----------
    CR : array-like
        2D or 1D array of critical ratio values.
    inj_CR : float
        Injected (or recovered) CR value to highlight.
    bins : int
        Number of histogram bins.
    window_sigma_factor : float
        Half-width of inset window in units of sigma.
    min_window_width : float
        Minimum half-width of inset window.
    figsize : tuple
        Figure size if ax is None.
    ax : matplotlib axis, optional
        If provided, draw into this axis.
    """

    x = CR.ravel()

    # --- Main histogram params (still can be dynamic) ---
    x_min = np.min(x)
    x_max_main = max(np.max(x), inj_CR)  # ensure inj visible even if out in tail
    bins = 100

    plt.figure()
    ax = plt.gca()

    # Use the SAME edges for main+inset so y-scaling is consistent
    edges = np.linspace(x_min, x_max_main, bins + 1)

    ax.hist(x, bins=edges, color='purple', alpha=0.8, edgecolor='k')
    ax.set_title(fr'$\mu$={np.mean(x):.4f}, $\sigma$={np.std(x):.3f}', fontsize=13)
    ax.set_xlabel('critical ratio', fontsize=14)
    ax.set_ylabel('count', fontsize=14)
    ax.grid(True, alpha=0.3)

    # --- Choose inset x-window around injection ---
    # window_halfwidth can scale with sigma or be a fixed fraction of inj_CR
    sigma = np.std(x)
    window_halfwidth = max(5.0, 2.5 * sigma)           # tweak factor if desired
    xlo = max(x_min, inj_CR - window_halfwidth)
    xhi = min(x_max_main, inj_CR + window_halfwidth)

    ax_inset = inset_axes(
        ax, width="38%", height="38%",
        loc='upper right',
        bbox_to_anchor=(-0.11, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    # --- Inset histogram using same edges, then set xlim to window ---
    n, _, _ = ax_inset.hist(x, bins=edges, color='purple', alpha=0.8, edgecolor='k')
    ax_inset.axvline(inj_CR, color='r', linestyle='--', linewidth=2)
    ax_inset.set_xlim(xlo, xhi+1)

    # --- Autoset inset y-limit based on bins within [xlo, xhi] ---
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (bin_centers >= xlo) & (bin_centers <= xhi)
    peak = n[mask].max() if np.any(mask) else n.max()
    ax_inset.set_ylim(0, max(1, peak * 1.15))

    # --- Connectors from window edges in main to inset bottom corners ---
    y0 = 0
    ax.add_artist(ConnectionPatch(
        xyA=(xlo, y0), coordsA=ax.transData,
        xyB=(0, 0), coordsB=ax_inset.transAxes,
        color='gray', linestyle='--', linewidth=1
    ))
    ax.add_artist(ConnectionPatch(
        xyA=(xhi, y0), coordsA=ax.transData,
        xyB=(1, 0), coordsB=ax_inset.transAxes,
        color='gray', linestyle='--', linewidth=1
    ))



