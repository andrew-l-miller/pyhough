from pathlib import Path
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np

from create_sfdbs.read_sfdb import sfdb_read_an_FFT
from create_sfdbs.convert_sciseg_file import load_sciseg_file, time_in_science

from pyhough import pm, physics, gfh, provider_injections
from pyhough import FFTing, time_conversions, inject, cand_sel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run transient CW injection and GFH recovery."
    )

    # analysis parameters
    parser.add_argument("--obs-run", default="O4a")
    parser.add_argument("--ifo", default="H1")
    parser.add_argument("--minf", type=float, default=800.0)
    parser.add_argument("--maxf", type=float, default=888.0)
    parser.add_argument("--max-num-ffts", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--fap", type=float, default=1e-3)
    parser.add_argument("--ref-perc-time", type=float, default=0.5)

    # injection parameters
    parser.add_argument("--mc", type=float, default=5.4409759e-4)
    parser.add_argument("--n", type=float, default=11 / 3)
    parser.add_argument("--f0", type=float, default=800.0)
    parser.add_argument("--h0", type=float, default=4.973e-23 / 2)

    # paths
    parser.add_argument(
        "--sfdb-dir",
        type=Path,
        default=Path("~/Downloads"),
        help="Directory containing .SFDB09 files.",
    )
    parser.add_argument(
        "--sciseg-file",
        type=Path,
        default=Path("~/Desktop/China/gwosc/gwosc/create_sfdbs/segsH1AnalysisReadyMinusVetoes_O4a_C00_g0f406df6.txt"),
        # required=True,
        help="Science-segment file to use.",
    )

    # flags
    parser.add_argument("--white-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--downsamp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--band", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gfh-nonuni", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--seed", type=int, default=5678)

    return parser.parse_args()




def run_tcw_injection(args, inj_provider=None):
    np.random.seed(args.seed)

    sfdb_dir = args.sfdb_dir.expanduser().resolve()
    sciseg_file = args.sciseg_file.expanduser().resolve()

    sfdb_files = sorted(sfdb_dir.glob("*.SFDB09"))

    if not sfdb_files:
        raise FileNotFoundError(f"No .SFDB09 files found in {sfdb_dir}")

    if not sciseg_file.exists():
        raise FileNotFoundError(f"Science-segment file not found: {sciseg_file}")

    sci_times = load_sciseg_file(sciseg_file)
    Nfil = len(sfdb_files)

    obs_run = args.obs_run
    ifo = args.ifo
    minf = args.minf
    maxf = args.maxf
    max_num_ffts = args.max_num_ffts
    threshold = args.threshold
    fap = args.fap
    ref_perc_time = args.ref_perc_time

    mc = args.mc
    n = args.n
    f0 = args.f0
    h0 = args.h0

    white_noise = args.white_noise
    downsamp = args.downsamp
    band = args.band
    plot = args.plot
    gfh_nonuni = args.gfh_nonuni

    ### derived parameters

    x0 = physics.get_x0_from_f0(f0,n)
    kn = physics.calc_k(mc)

    fdotmin = physics.calc_fdot_chirp(mc,minf) # calculate minimum fdot
    fdotmax = physics.calc_fdot_chirp(mc,maxf) # calculate maximum fdot
    t1 = physics.calc_time_to_coalescence(mc,minf) # time left to coalesence at minf
    t2 = physics.calc_time_to_coalescence(mc,maxf) # time left to coalesence at maxf
    dur = np.floor(t1-t2) # duration analyzed
    new_tfft = np.round(1/np.sqrt(fdotmax)) # confine all frequency modulations to 1 freq bin in each FFT


    ### choice of type of injection -- INSERT YOUR FUNCTION HERE

    # inj_provider = None
    # inj_provider = provider_injections.provider_sinusoid_drift(f0=860, fdot=0, h0=1e-22)
    inj_provider = provider_injections.provider_power_law(f0=f0,k=kn,n=n,h0=h0)


    if inj_provider is None:
        downsamp = False
        inj = None
    else:
        ant = inject.ligoh()
        nsid = 10000
        alpha,delta = inject.random_sky_deg()
        psi = inject.random_psi()
        cosiota = inject.random_cosiota()
        sour = inject.Source(alpha,delta,psi,cosiota)

        sid1 = np.real(inject.sidereal_lf_series(alpha, delta, 0.0, 0.0, ant, nsid))
        sid2 = np.real(inject.sidereal_lf_series(alpha, delta, 0.0, 45.0, ant, nsid))    
        ctx = provider_injections.InjContext(source=sour,sid1=sid1, sid2=sid2)

        inj = provider_injections.Injection(provider=inj_provider, ctx=ctx)


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
                    cut_inds = (allt0s <= t_fin)
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

def main():
    args = parse_args()
    run_tcw_injection(args)


if __name__ == "__main__":
    main()
