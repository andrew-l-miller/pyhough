from pathlib import Path
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np

from create_sfdbs.read_sfdb import sfdb_read_an_FFT
from create_sfdbs.convert_sciseg_file import load_sciseg_file, time_in_science

from pyhough import pm, physics, gfh, provider_injections
from pyhough import FFTing, time_conversions, inject, cand_sel,obs_runs

import json
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a transient-CW injection and GFH recovery test."
    )

    # ------------------------------------------------------------------
    # Data selection
    # ------------------------------------------------------------------

    parser.add_argument(
        "--obs-run",
        default="O4a",
        help="Observing run label used for bookkeeping and consistency checks.",
    )

    parser.add_argument(
        "--ifo",
        default="H1",
        choices=["H1", "L1"],
        help="Detector to analyze.",
    )

    parser.add_argument(
        "--sfdb-dir",
        type=Path,
        default=Path("~/Downloads"),
        help="Directory containing the .SFDB09 files to read.",
    )

    parser.add_argument(
        "--sciseg-file",
        type=Path,
        default=Path(
            "~/Desktop/China/gwosc/gwosc/create_sfdbs/"
            "segsH1AnalysisReadyMinusVetoes_O4a_C00_g0f406df6.txt"
        ),
        help="Science-segment file used to check whether the analyzed data are in science time.",
    )

    # ------------------------------------------------------------------
    # Frequency band and analysis duration
    # ------------------------------------------------------------------

    parser.add_argument(
        "--minf",
        type=float,
        default=800.0,
        help="Lower edge of the analyzed frequency band [Hz].",
    )

    parser.add_argument(
        "--maxf",
        type=float,
        default=888.0,
        help="Upper edge of the analyzed frequency band [Hz].",
    )

    parser.add_argument(
        "--max-num-ffts",
        type=int,
        default=100,
        help="Maximum number of original SFDB FFTs to process.",
    )

    parser.add_argument(
        "--dur",
        type=float,
        default=5000.,
        help="Nominal analyzed duration [s]. Currently superseded if duration is recomputed from the chirp model.",
    )

    parser.add_argument(
        "--tfft",
        type=float,
        default=4.0,
        help="Nominal output FFT duration [s]. Currently superseded if TFFT is recomputed from the chirp model.",
    )

    # ------------------------------------------------------------------
    # GFH / peakmap settings
    # ------------------------------------------------------------------

    parser.add_argument(
        "--threshold",
        type=float,
        default=2.5,
        help="Equalized-power threshold used to construct the peakmap.",
    )

    parser.add_argument(
        "--fap",
        type=float,
        default=1e-3,
        help="False-alarm probability fraction used to decide how many Hough-map candidates to retain.",
    )

    parser.add_argument(
        "--ref-perc-time",
        type=float,
        default=0.5,
        help="Reference time for the GFH map as a fraction of the analyzed duration "
             "(0=start, 0.5=middle, 1=end).",
    )

    # ------------------------------------------------------------------
    # Source / chirp model parameters
    # ------------------------------------------------------------------

    parser.add_argument(
        "--mc",
        type=float,
        default=5.4409759e-4,
        help="Chirp mass used to compute the power-law chirp evolution.",
    )

    parser.add_argument(
        "--n",
        type=float,
        default=11 / 3,
        help="Power-law index n used in the transient-CW chirp model.",
    )

    parser.add_argument(
        "--f0",
        type=float,
        default=800.0,
        help="Reference injected frequency [Hz].",
    )

    parser.add_argument(
        "--h0",
        type=float,
        default=4.973e-23 / 2,
        help="Injected strain amplitude.",
    )

    parser.add_argument(
        "--fdot",
        type=float,
        default=0.0,
        help="Frequency derivative [Hz/s], used by sinusoid_drift injections.",
    )

    # ------------------------------------------------------------------
    # Injection model
    # ------------------------------------------------------------------

    parser.add_argument(
        "--inj-provider",
        choices=list(provider_injections.PROVIDER_REGISTRY.keys()),
        default=None,
        help="Injection provider to use. If omitted, no software injection is performed.",
    )

    parser.add_argument(
        "--inj-kwargs",
        type=str,
        default="{}",
        help=(
            "JSON dictionary of keyword arguments passed to the injection provider. "
            "Values here override the defaults from --f0, --fdot, --h0, --mc, and --n. "
            "Example: '{\"f0\": 860, \"fdot\": 0, \"h0\": 1e-22}'"
        ),
    )

    # ------------------------------------------------------------------
    # Runtime flags
    # ------------------------------------------------------------------

    parser.add_argument(
        "--white-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use simulated white noise instead of the SFDB data.",
    )

    parser.add_argument(
        "--downsamp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downsample to the requested analysis band before constructing the time-frequency map.",
    )

    parser.add_argument(
        "--band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict the time-frequency map to the requested frequency band.",
    )

    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make diagnostic plots.",
    )

    parser.add_argument(
        "--gfh-nonuni",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the non-uniform GFH grid and empirical CR statistic.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Random-number seed used for injected sky position and orientation.",
    )

    return parser.parse_args()




def run_tcw_injection(args, inj_provider=None):
    np.random.seed(args.seed)

    sfdb_dir = args.sfdb_dir.expanduser().resolve()
    sciseg_file = args.sciseg_file.expanduser().resolve()

    obs_run = args.obs_run
    ifo = args.ifo

    sfdb_files = sorted(sfdb_dir.glob("*.SFDB09"))

    if not sfdb_files:
        raise FileNotFoundError(f"No .SFDB09 files found in {sfdb_dir}")

    if not sciseg_file.exists():
        raise FileNotFoundError(f"Science-segment file not found: {sciseg_file}")

    if ifo not in sciseg_file.name:
        raise ValueError(
            f"IFO '{ifo}' does not match science-segment file "
            f"'{sciseg_file.name}'"
    )

    sci_times = load_sciseg_file(sciseg_file)
    Nfil = len(sfdb_files)

    # obs_runs.check_science_segments_match_run(sci_times,obs_run)

    minf = args.minf
    maxf = args.maxf
    max_num_ffts = args.max_num_ffts
    threshold = args.threshold
    fap = args.fap
    ref_perc_time = args.ref_perc_time


    white_noise = args.white_noise
    downsamp = args.downsamp
    band = args.band
    plot = args.plot
    gfh_nonuni = args.gfh_nonuni

    
    provider_type = getattr(inj_provider, "provider_type", None)
    inj_kwargs = json.loads(args.inj_kwargs)

    if inj_provider is None and args.inj_provider is not None:

        # Add derived/default parameters only when needed and not already supplied
        if args.inj_provider == "power_law":

            mc = inj_kwargs.get("mc", args.mc)
            n = inj_kwargs.get("n", args.n)
            f0 = inj_kwargs.get("f0", args.f0)
            h0 = inj_kwargs.get("h0", args.h0)

            ### derived parameters


            x0 = physics.get_x0_from_f0(f0,n)
            kn = physics.calc_k(mc)

            inj_kwargs.setdefault("f0", f0)
            inj_kwargs.setdefault("h0", h0)
            inj_kwargs.setdefault("n", n)
            # inj_kwargs.setdefault("k", kn)

        elif args.inj_provider == "sinusoid_drift":
            inj_kwargs.setdefault("f0", f0)
            inj_kwargs.setdefault("h0", h0)
            inj_kwargs.setdefault("fdot", args.fdot)

        elif args.inj_provider == "sinusoid":
            inj_kwargs.setdefault("f0", f0)
            inj_kwargs.setdefault("h0", h0)

        provider_factory = provider_injections.PROVIDER_REGISTRY[args.inj_provider]
        inj_provider = provider_factory(**inj_kwargs)

    
    provider_type = getattr(inj_provider, "provider_type", args.inj_provider)
    provider_params = getattr(inj_provider, "params", {})

    mc = provider_params.get("mc", inj_kwargs.get("mc", args.mc))
    n = provider_params.get("n", inj_kwargs.get("n", args.n))
    f0 = provider_params.get("f0", inj_kwargs.get("f0", args.f0))
    h0 = provider_params.get("h0", inj_kwargs.get("h0", args.h0))

    x0 = physics.get_x0_from_f0(f0, n)
    kn = physics.calc_k(mc)

    if inj_provider is None:
        downsamp = False
        inj = None
    else:
        if ifo == 'H1':
            ant = inject.ligoh()
        elif ifo == 'L1':
            ant = inject.ligol()
        nsid = 10000
        alpha,delta = inject.random_sky_deg()
        psi = inject.random_psi()
        cosiota = inject.random_cosiota()
        sour = inject.Source(alpha,delta,psi,cosiota)

        sid1 = np.real(inject.sidereal_lf_series(alpha, delta, 0.0, 0.0, ant, nsid))
        sid2 = np.real(inject.sidereal_lf_series(alpha, delta, 0.0, 45.0, ant, nsid))    
        ctx = provider_injections.InjContext(source=sour,sid1=sid1, sid2=sid2)

        inj = provider_injections.Injection(provider=inj_provider, ctx=ctx)

        provider_type = getattr(inj_provider, "provider_type", None)
    
    has_mc = ("mc" in provider_params) or ("mc" in inj_kwargs)
    has_n  = ("n"  in provider_params) or ("n"  in inj_kwargs) 
    if has_mc and has_n :

        fdotmin = physics.calc_fdot_chirp(mc,minf) # calculate minimum fdot
        fdotmax = physics.calc_fdot_chirp(mc,maxf) # calculate maximum fdot
        t1 = physics.calc_time_to_coalescence(mc,minf) # time left to coalesence at minf
        t2 = physics.calc_time_to_coalescence(mc,maxf) # time left to coalesence at maxf

        dur = np.floor(t1-t2) # duration analyzed
        new_tfft = np.round(1/np.sqrt(fdotmax)) # confine all frequency modulations to 1 freq bin in each FFT
    else:
        fdotmin = None
        fdotmax = None
        dur = args.dur
        new_tfft = args.tfft

    print("duration (s): ", dur)
    print("TFFT (s): ",new_tfft)


    allt0s = []
    fft_index = 0
    for i in range(Nfil):
        sfdb_file = sfdb_files[i]
        with open(sfdb_file, "rb") as f:
            while True:
                sfdb_head, _, sps, sft = sfdb_read_an_FFT(f)
                if sfdb_head == 0:
                    break
                if (np.sum(sps) == 0) or (np.sum(np.abs(sft)) == 0):
                    continue
                if fft_index == 0:
                    t0 = sfdb_head.gps_sec
                    dt = sfdb_head.tsamplu
                    tfft = sfdb_head.tbase
                    nsam = sfdb_head.nsamples
                    df = sfdb_head.deltanu
                    N_FFT = np.ceil(dur / (tfft / 2))
                    t_fin = t0 + dur
                    _,_,t_in,_ = time_in_science(t0,t_fin,sci_times)

                    if white_noise == False:
                        if t_in / dur < 0.99:
                            continue
                        else:
                            print('signal will be completely in sci time')
            
                    
                    # norm_factor = sfdb_head.normd * sfdb_head.normw * np.sqrt(2)
                    # all_t0s_in_sfdb = np.arange(max_num_ffts)*tfft/2+t0
                    # vs = get_detector_velocities(all_t0s_in_sfdb,tfft,ifo)
                
                # if white_noise:
                #     lfft = int(tfft / dt)
                #     sft,sps = FFTing.sub_whitenoise(lfft,sfdb_head)


                times,freqs,FFTs, SPSs, tf_map = FFTing.change_FFT_length(sft,sfdb_head,new_tfft,minf,maxf,inj,fft_index,downsamp,band,white_noise)
                if band or downsamp:
                    freqs = freqs + minf
                if fft_index == 0:
                    whole_map = tf_map.copy()
                else:
                    whole_map = np.concatenate((whole_map, tf_map), axis=1)

                allt0s.extend(times)
                
                print(f"[{fft_index+1}/{int(N_FFT+1)}] FFTs complete.")

                fft_index += 1
                if fft_index > N_FFT:
                    cut_inds = np.asarray(allt0s) <= t_fin
                    allt0s = np.asarray(allt0s)[cut_inds]
                    whole_map = whole_map[:,cut_inds]
                    break


    ### Create peakmap
    pm_times,pm_freqs,pm_pows,index = pm.make_peakmap_from_spectrogram(allt0s,freqs,whole_map,threshold)

    Nfs = len(np.arange(minf, maxf, 1/new_tfft))

    weights = np.ones(len(pm_freqs)) #### could eventually make adaptive

    ### calculate empirical probability of selecting a peak
    if gfh_nonuni:
        p0emp = pm.calc_p0_empirical(weights,index,Nfs)



    if band or downsamp:
        in_band = np.ones(pm_freqs.size, dtype=bool)
    else:
        in_band = (pm_freqs > minf) & (pm_freqs<maxf)

    if plot:
        pm.python_plot_triplets((pm_times[in_band]-pm_times[0]),pm_freqs[in_band],(pm_pows[in_band]),'.',label='equalized power')

        if not band:
            fig2,ax2 = plt.subplots()
            FFTing.test_exp_dist(whole_map[(freqs > minf) & (freqs<maxf),:].flatten())
        else:
            fig3,ax3 = plt.subplots()
            FFTing.test_exp_dist(whole_map.flatten())

    ### construct grid in k to do the GFH

    if fdotmin is None or fdotmax is None:
        print(
            "No chirp model supplied. "
            "Peakmap generation completed; GFH search skipped."
        )
        return

    gridk,dk = gfh.andrew_long_transient_grid_k(new_tfft,[minf, maxf],[fdotmin, fdotmax],dur,n)
    gridk = np.squeeze(gridk)
    # gridk = np.squeeze(gfh.cbc_shorten_gridk(gridk,sour['kn'],sour['kn']))

    t00_ref_time = allt0s[0] + dur * ref_perc_time
    epoch = time_conversions.gps2mjd(t00_ref_time)

    hm_job = gfh.make_hm_job_struct(minf,maxf,new_tfft,dur,n,ref_perc_time,gridk,epoch)

    p = np.array([
        time_conversions.gps2mjd(pm_times[in_band]),
        pm_freqs[in_band],
        pm_pows[in_band]
    ])


    ### Run the GFH

    if gfh_nonuni:
        hmap,info = gfh.LongT_GENERALIZED_fasthough_nonuni(p,hm_job)

        times_mjd = time_conversions.gps2mjd(allt0s)

        ### Estimate empirically the number of peaks on average to fall into each bin in Hough map, and standard deviation

        MU, ST = gfh.vary_p0_hfdf_compute_mu_sigma_nonuni_grids(hm_job['gridx'], gridk, hm_job['n'], new_tfft, minf,maxf,times_mjd, hm_job['epoch'], p0emp)
        info['MU'] = MU
        info['ST'] = ST
        CR = (hmap - MU.T) / ST.T
    else:        
        hmap,info = gfh.LongT_GENERALIZED_fasthough(p,hm_job)

    if plot:
        if gfh_nonuni:
            gfh.plot_hm(CR,info,physical=True,lab='CR')
        else:
            gfh.plot_hm(hmap,info,physical=True)



    nk,nx = hmap.shape
    kcand = int(np.floor(nk * nx * fap)) ### number of candidates to select in the hough map
    if gfh_nonuni:
        cand2,more_info = cand_sel.select_cands_transients_nonuni(CR,info,kcand)
    else:
        pass
        # cand2,more_info = cand_sel.hfdf_peak_transients(hmap,info,int(np.sqrt(kcand)),deltaf=0,dn=0)
    ### select significant candidates in the Hough map
    cand2 = cand2[:,np.any(cand2 != 0, axis=0)]

    ### Since the hough map is created at reference time epoch, (1/2 through Tobs), need to shift source parameters to that time

    offset = t00_ref_time - allt0s[0] 
    xnew = physics.shift_x0_by_time(x0,kn,offset,n)

    ### Determine if the source was found

    found, best_cand, mindist, dist, dist_each_parm_best_inj = cand_sel.coin_inj_cand(cand2,xnew,kn,coin_dist=3)


    ### plot a histogram of the CR background and injection foreground

    if plot and gfh_nonuni:
        cand_sel.plot_CR_histogram_with_inset(CR,best_cand[5])


    print("DONE")


# =============================================================================
# Example configurations
# =============================================================================
# Uncomment ONE block at a time.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. Run the search with no injection
# -----------------------------------------------------------------------------
# Uses the SFDB data directly and runs the standard GFH analysis.

# sys.argv = [
#     "run_tcw_injection.py",
#     "--minf", "850",
#     "--maxf", "860",
# ]


# -----------------------------------------------------------------------------
# 2. Inject a built-in sinusoid_drift signal
# -----------------------------------------------------------------------------
# Uses a provider registered in PROVIDER_REGISTRY.
# Useful for quick tests without modifying the code.

# sys.argv = [
#     "run_tcw_injection.py",
#     "--minf", "850",
#     "--maxf", "860",
#     "--inj-provider", "sinusoid_drift",
#     "--inj-kwargs", '{"f0": 855, "fdot": 0, "h0": 1e-22}',
# ]


# -----------------------------------------------------------------------------
# 3. Inject a built-in power-law chirp signal
# -----------------------------------------------------------------------------
# Uses the default CBC-like transient CW model.

# sys.argv = [
#     "run_tcw_injection.py",
#     "--minf", "800",
#     "--maxf", "888",
#     "--inj-provider", "power_law",
#     "--inj-kwargs", '{"f0": 800, "h0": 2.4865e-23, "mc": 3e-4}',
# ]


# -----------------------------------------------------------------------------
# 4. Inject a custom signal provider
# -----------------------------------------------------------------------------
# Use this when testing a new provider without adding it to
# PROVIDER_REGISTRY or modifying parse_args().
#
# The custom provider is passed directly to run_tcw_injection().

# sys.argv = [
#     "run_tcw_injection.py",
#     "--minf", "850",
#     "--maxf", "860",
# ]

# args = parse_args()

# my_provider = provider_injections.provider_sinusoid_drift(
#     f0=855,
#     fdot=0,
#     h0=1e-22,
# )

# run_tcw_injection(args, inj_provider=my_provider)
#
# sys.exit()


# -----------------------------------------------------------------------------
# 5. Inject a power-law signal provider
# -----------------------------------------------------------------------------
# Use this when testing a new provider without adding it to
# PROVIDER_REGISTRY or modifying parse_args().
#
# The custom provider is passed directly to run_tcw_injection().


sys.argv = [
    "run_tcw_injection.py",
    "--minf", "800",
    "--maxf", "888",
]

args = parse_args()

my_provider = provider_injections.provider_power_law(
    f0=800,
    h0=2.4865e-23,
    mc=4e-4,
    n=11/3,
)

run_tcw_injection(args, inj_provider=my_provider)


# =============================================================================
# Normal execution path
# =============================================================================

def main():
    args = parse_args()
    run_tcw_injection(args)


if __name__ == "__main__":
    main()