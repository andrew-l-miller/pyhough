import numpy as np
from pyhough.pm import *

def hfdf_hough_spectrogram(times,freqs,power,Tsft,sdgrid,threshold,vec_n,vels,ref_perc_time):
#### times: unique times array
#### freqs: unique freqs array
#### power: spectrogram of size len(freqs) x len(times)
    inif = np.min(freqs)
    finf = np.max(freqs)
    df = 1/Tsft
    Nsds = len(sdgrid)
    Ntts = len(times)

    Nf0s = int(np.ceil((finf-inif)/df))

    binh_df0 = np.zeros((Nsds,Nf0s))
    t0 = np.percentile(times,ref_perc_time)

    times = times - t0
    for t in range(Ntts):
        inds_above_thr = np.array(select_local_max(power[:,t],threshold))
        if inds_above_thr.shape[0] == 0:
            continue
        
        freqs_peaks = freqs[inds_above_thr]
        freqs_dopp_corr = remove_doppler(freqs_peaks,vec_n,vels[:,t])
        these_fs = ( freqs_dopp_corr - inif ) / df ;
        tt_df = times[t] / df
        for k in range(Nsds):
            td = sdgrid[k] * tt_df
            inds = np.round(these_fs - td).astype(int) ### f = f0 + fdot(t-t0) solve for f0, f0 = f - fdot(t-t0)
            ind_of_inds = np.argwhere(inds>=0);
            a = inds[ind_of_inds];
            log_inds = a <= Nf0s-1;
            a = a[log_inds];
            binh_df0[k,a] = binh_df0[k,a] + 1
    
    return binh_df0


def hfdf_hough(times,peak_freqs,Tsft,sdgrid,ref_perc_time):

    inif = np.min(peak_freqs)
    finf = np.max(peak_freqs)
    df = 1/Tsft
    Nsds = len(sdgrid)
    Nf0s = int(np.ceil((finf-inif)/df))

    binh_df0 = np.zeros((Nsds,Nf0s))
    t0 = np.percentile(times,ref_perc_time)

    ii0 = 0
    ii = np.squeeze(np.argwhere(np.diff(times)))
    times = times - t0
    Ntts = len(ii)
    for t in range(Ntts):
        these_fs = ( peak_freqs[ii0:ii[t]+1] - inif ) / df 
        tt_df = times[ii0] / df
        for k in range(Nsds):
            td = sdgrid[k] * tt_df
            inds = np.round(these_fs - td).astype(int) ### f = f0 + fdot(t-t0) solve for f0, f0 = f - fdot(t-t0)
            ind_of_inds = np.argwhere(inds>=0);
            a = inds[ind_of_inds];
            log_inds = a <= Nf0s-1;
            a = a[log_inds];
            binh_df0[k,a] = binh_df0[k,a] + 1
            
            
        ii0 = ii[t]+1
    return binh_df0

def make_sd_grid(sig_fdot,dsd):
    min_fdot_search = sig_fdot * 10
    max_fdot_search = np.abs(sig_fdot * 10)
    if min_fdot_search == 0:
        min_fdot_search = -10 * dsd
        max_fdot_search = 10 * dsd
    sdgrid = np.arange(min_fdot_search,max_fdot_search,dsd)
    return sdgrid
