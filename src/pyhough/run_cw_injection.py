

import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
from pyhough import pm,hm

from create_sfdbs.read_sfdb import sfdb_read_an_FFT
from create_sfdbs.convert_sciseg_file import load_sciseg_file,time_in_science


from pyhough.provider_injections import InjContext,provider_cw,provider_sinusoid,provider_sinusoid_drift
from pyhough.inject import Source,ligoh,ligol,phase_from_frequency,inject_sig_into_sft,random_cosiota,random_psi,random_sky_deg,sidereal_lf_series,get_detector_velocities
from pyhough.FFTing import test_exp_dist, calc_dsfact, get_sft_sps_and_f_inds,get_downsampled_times_samps,sub_whitenoise
from pyhough.time_conversions import gps2mjd,gmst
import argparse
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------
# Data selection
# ------------------------------------------------------------------

    parser.add_argument(
        "--obs-run",
        default="O4a",
        help="Observing run label used for bookkeeping and output naming.",
    )

    parser.add_argument(
        "--ifo",
        default="H1",
        choices=["H1", "L1"],
        help="Detector to analyze (H1 or L1).",
    )

    parser.add_argument(
        "--sfdb-glob",
        default="/Users/andrewmiller/Downloads/*.SFDB09",
        help="Glob pattern specifying the SFDB files to read.",
    )

    parser.add_argument(
        "--sciseg-file",
        type=Path,
        default=Path(
            "/Users/andrewmiller/Desktop/China/gwosc/gwosc/create_sfdbs/"
            "segsH1AnalysisReadyMinusVetoes_O4a_C00_g0f406df6.txt"
        ),
        help="Science-segment file used to define valid observing times.",
    )

    parser.add_argument(
        "--white-noise", 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help = "white noise or not; true if yes; false if no",
        )


    parser.add_argument(
        "--doppler-correct",
        action="store_true",
        default=False,
        help="Apply Doppler correction to the peakmap using alpha/delta.",
    )

    parser.add_argument(
        "--run-hough",
        action="store_true",
        default=False,
        help="Run the spin-down Hough transform instead of projecting peaks.",
    )



    # ------------------------------------------------------------------
    # Frequency band
    # ------------------------------------------------------------------

    parser.add_argument(
        "--minf",
        type=float,
        default=1205.0,
        help="Lower edge of the analyzed frequency band [Hz].",
    )

    parser.add_argument(
        "--band",
        type=float,
        default=1.0,
        help="Width of the analyzed frequency band [Hz]. "
            "The upper edge is minf + band.",
    )

    # ------------------------------------------------------------------
    # Injection model
    # ------------------------------------------------------------------

    parser.add_argument(
        "--inj-provider",
        choices=["cw", "sinusoid", "sinusoid_drift"],
        default=None,
        help="Signal model used for the software injection.",
    )

    # ------------------------------------------------------------------
    # Signal parameters
    # ------------------------------------------------------------------

    parser.add_argument(
        "--h0",
        type=float,
        default=2e-24,
        help="Injected strain amplitude.",
    )

    parser.add_argument(
        "--f0",
        type=float,
        default=None,
        help="Injected signal frequency [Hz]. "
            "If not supplied, the center of the search band is used.",
    )

    parser.add_argument(
        "--fdot",
        type=float,
        default=0.0,
        help="Injected first frequency derivative [Hz/s].",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=50.0,
        help="Injected right ascension in degrees.",
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=-20.0,
        help="Injected declination in degrees.",
    )

    parser.add_argument(
        "--inj-kwargs",
        type=json.loads,
        default={},
        help="JSON dictionary of keyword arguments passed to the injection provider.",
    )

    # ------------------------------------------------------------------
    # Analysis settings
    # ------------------------------------------------------------------

    parser.add_argument(
        "--threshold",
        type=float,
        default=2.5,
        help="Equalized-power threshold used when constructing the peakmap.",
    )

    parser.add_argument(
        "--ref-perc-time",
        type=float,
        default=0.5,
        help="Reference time used for Hough spin-down corrections, "
            "expressed as a fraction of the observation span "
            "(0=start, 0.5=middle, 1=end).",
    )

    parser.add_argument(
        "--downsamp",
        action="store_true",
        default=True,
        help="Inject signals using the downsampled narrow-band workflow.",
    )

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    parser.add_argument(
        "--max-num-ffts",
        type=int,
        default=100,
        help="Maximum number of FFTs used when constructing auxiliary "
            "velocity and timing arrays.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Random-number seed used for source orientation and sky draws.",
    )

    return parser.parse_args()


def run_cw_injection(args,inj_provider = None):

    if inj_provider is not None:
        print("custom injection")
        custom = True

    sciseg_file = args.sciseg_file
    sci_times = load_sciseg_file(sciseg_file)

    sfdb_files = glob.glob(args.sfdb_glob)
    if not sfdb_files:
        raise FileNotFoundError("No .SFDB09 files found in the current directory.")
    else:
        sfdb_file = sfdb_files[0]
        print(f"Using SFDB file: {sfdb_file}")
    minf = args.minf
    maxf = minf + args.band
    # maxf = minf + 1

    allt0s = []
    spec_rows = []   # list of 1D arrays (each is eq_power or band-cut)
    all_vsss = []   # will become (3, N)
    all_psss = []
    fft_index = 0
    downsamp = args.downsamp

    max_num_ffts = args.max_num_ffts

    fmean = 0.5 * (maxf + minf)

    f0_inj = args.f0 if args.f0 is not None else fmean
    fdot_inj = args.fdot

    position = [args.alpha, args.delta]

    if inj_provider is None and args.inj_provider is not None:
        inj_kwargs = dict(args.inj_kwargs)

        inj_kwargs.setdefault("f0", f0_inj)
        inj_kwargs.setdefault("fdot", fdot_inj)
        inj_kwargs.setdefault("h0", args.h0)

        if args.inj_provider == "cw":
            inj_kwargs.setdefault("alpha", args.alpha)
            inj_kwargs.setdefault("delta", args.delta)

        f0_inj = inj_kwargs["f0"]
        fdot_inj = inj_kwargs.get("fdot", 0.0)

        position = [
            inj_kwargs.get("alpha", args.alpha),
            inj_kwargs.get("delta", args.delta),
        ]

        if args.inj_provider == "cw":
            inj_provider = provider_cw(**inj_kwargs)

        elif args.inj_provider == "sinusoid":
            inj_provider = provider_sinusoid(**inj_kwargs)

        elif args.inj_provider == "sinusoid_drift":
            inj_provider = provider_sinusoid_drift(**inj_kwargs)

        else:
            raise ValueError(f"Unknown injection provider: {args.inj_provider}")

            # cw = inj_provider.__qualname__.__contains__('provider_cw') ### True if CW; false otherwise
    elif inj_provider is not None:
        print("custom injection provided")
    else:
        print("no injection")
    

    # alpha,delta = random_sky_deg()
    if inj_provider is not None:
        if args.ifo == 'H1':
            ant = ligoh()
        elif args.ifo == 'L1':
            ant = ligol()
        nsid=10000
        psi = random_psi()
        cosiota = random_cosiota()
        sour = Source(position[0],position[1],psi,cosiota)
        sid1 = np.real(sidereal_lf_series(position[0], position[1], 0.0, 0.0, ant, nsid))
        sid2 = np.real(sidereal_lf_series(position[0], position[1], 0.0, 45.0, ant, nsid))
        SD = 86164.09053083288 #sidereal day
    else:
        print("no sidereal modulation or sour structure simulated")


    for sfdb_file in sfdb_files:
        print(f"Processing {sfdb_file}")
        with open(sfdb_file, "rb") as f:
            while True:
                sfdb_head, _, sps, sft = sfdb_read_an_FFT(f)
                if sfdb_head == 0:
                    break
                if fft_index == 0:
                    t0 = sfdb_head.gps_sec
                    dt = sfdb_head.tsamplu
                    tfft = sfdb_head.tbase
                    nsam = sfdb_head.nsamples
                    df = sfdb_head.deltanu
                    red = sfdb_head.red

                    
                    freqs = np.arange(nsam) * df
                    times = np.arange(nsam) * dt * 2 ## times 2 b/c simulating complex signal --> f_nyg = f_samp
                    norm_factor = sfdb_head.normd * sfdb_head.normw * np.sqrt(2)
                    all_t0s_in_sfdb = np.arange(max_num_ffts)*tfft/2+t0
                    vs = get_detector_velocities(all_t0s_in_sfdb,tfft,args.ifo)
        
                    if inj_provider is not None:
                        ctx = InjContext(vs=vs,sid1=sid1,sid2=sid2,source=sour)
                    k1, k2, fr1, fr2, *_ = get_sft_sps_and_f_inds(minf, maxf, nsam, df, red, dsfact=1.0)

                if args.white_noise:
                     sft,sps = sub_whitenoise(nsam,sfdb_head.normd,sfdb_head.normw)
                     sps_full = sps / norm_factor
                else:
                    sps_full = np.repeat(sps, red) / norm_factor
                
                if inj_provider is not None:
                    mjd_time = gps2mjd(sfdb_head.gps_sec)

                    if downsamp:
                        # Downsampled injection uses only the [k1:k2] band
                        dsfact, NORM = calc_dsfact(dt, minf, maxf)

                        sft_band = sft[k1:k2]
                        nfftnew = k2 - k1

                        # dtnew determined by your helper (must match your change_FFT_length logic)
                        dtnew, _ = get_downsampled_times_samps(dt, nsam, dsfact, sft_band)

                        inj_times = np.arange(nfftnew) * dtnew

                        # sidereal time samples (hours) on SAME grid
                        st = gmst(mjd_time) + dtnew * (86400.0 / SD) * np.arange(nfftnew) / 3600.0

                        # absolute time used by phase model
                        tt = inj_times + (sfdb_head.gps_sec - t0)

                        amps, fsss = inj_provider(tt, fft_index, ctx)

                        # match change_FFT_length downsamp branch: band-relative freq
                        fsss = fsss - fr1

                        phase_evol = phase_from_frequency(tt, fsss)

                        i1 = np.mod(np.floor(st * (nsid - 1) / 24.0 + 0.5).astype(int), nsid - 1)
                        sid1_t = ctx.sid1[i1]
                        sid2_t = ctx.sid2[i1]

                        sig_t = amps * (sour.Hp() * sid1_t + sour.Hc() * sid2_t) * np.exp(1j * phase_evol)

                        # inject into band only
                        sft[k1:k2] = inject_sig_into_sft(sig_t, sft_band, NORM)

                    else:
                        # Non-downsampled injection uses the full stored one-sided SFT
                        # Your convention: analytic signal sampled at 2*dt with length nsam (= nfft/2)
                        NORM = 1.0

                        st = gmst(mjd_time) + dt * (86400.0 / SD) * np.arange(nsam) / 3600.0

                        tt = times + (sfdb_head.gps_sec - t0)

                        amps, fsss = inj_provider(tt, fft_index, ctx)
                        phase_evol = phase_from_frequency(tt, fsss)

                        i1 = np.mod(np.floor(st * (nsid - 1) / 24.0 + 0.5).astype(int), nsid - 1)
                        sig_t = amps * (sour.Hp() * ctx.sid1[i1] + sour.Hc() * ctx.sid2[i1]) * np.exp(1j * phase_evol)

                        # inject into full SFT, then (optionally) slice later when building the band spectrogram
                        sft = inject_sig_into_sft(sig_t, sft, NORM)
            
                if (np.sum(sps_full) == 0) or (np.sum(np.abs(sft)) == 0):
                    eq_power = np.zeros((np.shape(sps_full)))
                else:
                    eq_power = np.abs(sft)**2 / sps_full ** 2
                
                in_band = (freqs >= minf) & (freqs <= maxf)
                spec_rows.append(eq_power[in_band])
                
                allt0s.append(sfdb_head.gps_sec)
                fft_index += 1

            

    spec = np.vstack(spec_rows).T  # shape: (nfreq_in_band, ntime)
    allt0s = np.array(allt0s)
    band_freqs = freqs[in_band]
    t_rel = allt0s - allt0s[0]

    threshold = args.threshold
    pm_times,pm_freqs,pm_pows,index = pm.make_peakmap_from_spectrogram(allt0s,band_freqs,spec,threshold)


    pm.python_plot_triplets((pm_times-pm_times[0])/86400,pm_freqs,pm_pows,'.',label='equalized power')
    # plt.ylim(1205.3,1205.7)

    Nts = len(allt0s)
    vec_n = pm.astro2rect(position,1)
    vs = get_detector_velocities(allt0s,tfft,args.ifo)
    if args.doppler_correct: ## covers boson-cloud and neutron-star CWs
        freqs_new=pm.remove_doppler_from_peakmap(pm_times,pm_freqs,index,vec_n,vs.T,Nts)
    else: ## covers dark-matter sinusoid signal, when fdot = 0
        freqs_new=pm_freqs

    pm.python_plot_triplets((pm_times-pm_times[0])/86400,freqs_new,pm_pows,'.',label='equalized power')
    # plt.ylim((1205.4,1205.6))


    df = 1/tfft # the frequency bin size
    
    if args.run_hough:
        Tobs = np.max(allt0s-allt0s[0]) # the duration of the peakmap in seconds
        ref_perc_time = args.ref_perc_time # reference time for the Hough at which f0 is determined, set any number between 0 (beginning), 100 (end)
        dsd = 1/(tfft*Tobs) # step in spin-down: dsd = df / Tobs
        min_fdot_search = -10 * dsd
        max_fdot_search = 10 * dsd 
        sdgrid = np.arange(min_fdot_search,max_fdot_search,dsd)

        t00_ref_time = allt0s[0] + Tobs * ref_perc_time

        hmap = hm.hfdf_hough(pm_times,freqs_new,tfft,sdgrid,t00_ref_time)
        fs_for_hmap_from_pm = np.arange(np.min(freqs_new),np.max(freqs_new),df)

        if inj_provider is not None and custom is False:
            fnew = f0_inj + fdot_inj * Tobs * ref_perc_time
        else:
            fnew = None
        hm.plot_hm(fs_for_hmap_from_pm,sdgrid,hmap,fnew)
    else: ## boson cloud case ; dark photon case
        fbins,counts = pm.project_peaks(pm_freqs,freqs_new)
        plt.figure()
        plt.plot(fbins,counts)


    plt.figure()
    test_exp_dist(spec.flatten()[spec.flatten()>0],500)

    print("done")


# =============================================================================
# Example test configurations
# =============================================================================
# These sys.argv blocks are intended for interactive testing/debugging.
# Uncomment exactly one block at a time.
#
# Notes:
#   --doppler-correct applies sky-position Doppler correction using alpha/delta.
#   --run-hough runs the spin-down Hough transform.
#   If --run-hough is omitted, the code uses project_peaks().
#   If --inj-provider is omitted, no injection is added.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. Custom injection provider
# -----------------------------------------------------------------------------
# Use this when you want to define the injection provider manually without adding
# a new option to parse_args(). The command-line f0/fdot/alpha/delta values are
# still used as metadata for plotting, Doppler correction, and Hough markers.
# You will still have to put your custom injection provider in the provider_injections.py
# file.

sys.argv = [
    "run_cw_injection.py",
    "--minf", "859.5",
    "--band", "1.0",
    "--f0", "860",
    "--fdot", "1e-7",
    "--alpha", "30.0",
    "--delta", "-20.0",
    # "--doppler-correct",
    "--run-hough",
]

args = parse_args()

my_provider = provider_sinusoid_drift(
    f0=860,
    fdot=1e-7,
    h0=1e-23,
)

run_cw_injection(args, inj_provider=my_provider)


# -----------------------------------------------------------------------------
# 2. Built-in sinusoid-drift injection
# -----------------------------------------------------------------------------
# Injects a drifting sinusoid using the built-in provider_sinusoid_drift.
# No Doppler correction is applied unless --doppler-correct is included.

# sys.argv = [
#     "run_cw_injection.py",
#     "--minf", "859.5",
#     "--band", "1.0",
#     "--inj-provider", "sinusoid_drift",
#     "--inj-kwargs", '{"f0": 860, "fdot": 1e-7, "h0": 1e-23}',
# ]


# -----------------------------------------------------------------------------
# 3. Built-in CW injection with Doppler correction and Hough search
# -----------------------------------------------------------------------------
# Use this for neutron-star-like CW injections.
# For a boson-cloud-like monochromatic signal, set "fdot": 0 and omit --run-hough
# if you want project_peaks() instead of the Hough transform.

# sys.argv = [
#     "run_cw_injection.py",
#     "--minf", "859.5",
#     "--band", "1.0",
#     "--inj-provider", "cw",
#     "--inj-kwargs", '{"f0": 860, "fdot": 1e-7, "h0": 1e-23, "alpha": 30, "delta": -20}',
#     "--doppler-correct",
#     "--run-hough",
# ]


# =============================================================================
# No-injection searches
# =============================================================================


# -----------------------------------------------------------------------------
# 4. Boson-cloud-style search
# -----------------------------------------------------------------------------
# No injection. Applies Doppler correction at the requested sky position and then
# projects the corrected peakmap using project_peaks().

# sys.argv = [
#     "run_cw_injection.py",
#     "--minf", "859.5",
#     "--band", "1.0",
#     "--alpha", "30.0",
#     "--delta", "-20.0",
#     "--doppler-correct",
# ]


# -----------------------------------------------------------------------------
# 5. CW Hough search
# -----------------------------------------------------------------------------
# No injection. Applies Doppler correction and runs the spin-down Hough transform.

# sys.argv = [
#     "run_cw_injection.py",
#     "--minf", "859.5",
#     "--band", "1.0",
#     "--alpha", "30.0",
#     "--delta", "-20.0",
#     "--doppler-correct",
#     "--run-hough",
# ]


# -----------------------------------------------------------------------------
# 6. ULDM/instrument-coupled sinusoid search
# -----------------------------------------------------------------------------
# No injection. No Doppler correction. Uses project_peaks() directly.

# sys.argv = [
#     "run_cw_injection.py",
#     "--minf", "859.5",
#     "--band", "1.0",
# ]

def main():
    args = parse_args()
    run_cw_injection(args)


if __name__ == "__main__":
    main()