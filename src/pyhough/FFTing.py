
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from pyhough import inject
from pyhough import time_conversions

def pswindow(typ, length, par=None):
    """
    Computes windows for power spectrum estimates.
    
    Args:
        typ : int or str
            Window type:
            0 or 'no'       : no window (flat)
            1 or 'bartlett' : Bartlett window
            2 or 'hanning'  : Hanning window
            3 or 'flatcos'  : flat-top with cosine edge
            4 or 'tukey'    : Tukey window
            5 or 'gauss'    : Gaussian (3 sigma)
        length : int
            Length of the window
        par : float, optional
            Parameter for Tukey window (alpha)
    
    Returns:
        y : np.ndarray
            Window array of shape (length,)
    """
    y = np.ones(length)
    len2 = length // 2
    len4 = int(np.ceil(length / 4))

    # Map numeric types to strings
    if isinstance(typ, (int, float)):
        if typ == 1:
            typ = 'bartlett'
        elif typ == 2:
            typ = 'hanning'
        elif typ == 3:
            typ = 'flatcos'
        elif typ == 4:
            typ = 'tukey'
        elif typ == 5:
            typ = 'gauss'
        else:
            typ = 'no'

    typ = str(typ).lower()

    if typ == 'bartlett':
        y[:len2] = np.arange(1, len2+1) / len2
        y[len2:] = y[len2-1::-1]
        y *= np.sqrt(3)

    elif typ == 'hanning':
        y[:len2] = 1 - np.cos(np.arange(1, len2+1) * np.pi / len2)
        y[len2:] = y[len2-1::-1]
        y *= np.sqrt(2/3)

    elif typ == 'flatcos':
        y[:len4] = (1 - np.cos(np.arange(1, len4+1) * np.pi / len4)) / 2
        y[-len4:] = y[:len4][::-1]
        y *= np.sqrt(length / np.sum(y**2))

    elif typ == 'tukey':
        if par is None:
            par = 0.5  # default alpha if not provided
        lenx = int(round(par * length / 2))
        y[:lenx] = (1 - np.cos(np.arange(1, lenx+1) * np.pi / lenx)) / 2
        y[-lenx:] = y[:lenx][::-1]
        y *= np.sqrt(length / np.sum(y**2))

    elif typ == 'gauss':
        dx = 6 / (length - 1)
        x = np.arange(-3, 3 + dx, dx)
        y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
        y *= np.sqrt(length / np.sum(y**2))

    elif typ == 'no':
        y = np.ones(length)

    else:
        raise ValueError(f"Unknown window type: {typ}")

    return y


def get_original_strains(sft):
    opposite = np.flipud(sft)

    appending2 = np.hstack((0, np.conj(opposite[0:-1])))

    old_sft = np.hstack((sft, appending2))

    strains=np.fft.ifft(old_sft)
    
    return strains

def change_FFT_length(sft,sfdb_head,TFFT,minf,maxf,inj_provider=None,num_orig_FFT=0,downsamp=False,band=False,white_noise=False,win=3):
    
    strains =  get_original_strains(sft)
    
    nsamps = len(strains)
    dt = sfdb_head.tsamplu
    tfft_orig = sfdb_head.tbase
    red = sfdb_head.red
    

    nfft = int(TFFT / dt)
    num1 = nsamps // 4
    num2 = 3 * num1
    strains_to_fft = strains[num1:num2]

    n1 = len(strains_to_fft)
    nover = nfft // 2

    df = 1.0 / TFFT
    Fs = 1.0 / dt
    w = pswindow(win, nfft)  
    freqs = np.arange(0, Fs/2, df)
    
    ind = 0
    SD = 86164.09053083288 #sidereal day
    nsid = 10000 #### hardcoded to match when sid1 and sid2 are made

    num_FFTs = int(np.ceil(2 * n1 / nfft))
    all_t0s = []

    if (2 * n1) % nfft == 0:
        cond2 = num_FFTs + 10000
    else:
        cond2 = num_FFTs - 2

    k1,k2,fr1,fr2,kss1,kss2,_,_ = get_sft_sps_and_f_inds(minf,maxf,nfft // 2,df,red,1)
    if band:
        nfftnew = k2 - k1
        num_samps = nfftnew
    else:
        num_samps = nfft // 2

    all_FFTs = np.zeros((num_FFTs,num_samps), dtype=complex)
    all_SPSs = np.zeros_like(all_FFTs, dtype=float)
    tf_map = np.zeros_like(all_FFTs, dtype=float)

    gps0 = sfdb_head.gps_sec

    for jj in range(num_FFTs):
        if ind == 0:
            x = strains_to_fft[:nfft]
        elif ind == num_FFTs - 1 or ind == cond2:
            x = strains_to_fft[ind*nover:]
            if (2*n1) % nfft == 0:
                x = np.concatenate((x, np.zeros(nfft//2)))
            else:
                x = np.concatenate((x, np.zeros(nfft - len(x))))
        else:
            x = strains_to_fft[ind*nover : ind*nover + nfft]

        x = x * w
        xx = np.fft.fft(x)
        full_sft = np.sqrt(2) * xx[:nfft//2] ### sqrt(2) is bilat to unilat parameter, coming from neglecting neg freqs
        new_gps_sec = gps0 + (1/2) * ind * TFFT
        mjd_time = time_conversions.gps2mjd(new_gps_sec)
        if white_noise:
            normd = np.sqrt(dt/nfft)
            normw = sfdb_head.normw
            full_sft,full_sps = sub_whitenoise(nfft // 2,normd,normw)
            norm_factor = normd * normw * np.sqrt(2) 
        else:
            full_sps = np.sqrt(medfilt(np.abs(full_sft)**2, kernel_size=21)) / np.log(2)

        if inj_provider is not None:
            if jj == 0:
                sour = inj_provider.ctx.source

            if downsamp:
                dsfact,NORM = calc_dsfact(dt,minf,maxf)
                nsamp_orig = len(full_sft)
                dtnew,_  = get_downsampled_times_samps( dt,nsamp_orig,dsfact,full_sft[k1:k2] )
                inj_times = np.arange(nfftnew) * dtnew     
                st = time_conversions.gmst(mjd_time) + dtnew * (86400.0 / SD) * np.arange(nfftnew) / 3600.0

            else:
                NORM = 1.0  #### neeeds to be 1, because in sinusoid case, can recover amp h0 with np.max(np.abs((sig_f)) / len(sig_f)
                            #### 1 is consistent with how the inj_times are sampled in the non-downsampled case
                if not band:
                    k1 = 0
                    k2 = int(nfft /2)
                
                inj_times = np.arange(nfft / 2) * dt * 2 ### nfft/2 b/c nfft is including neg freq samples; analyt sig sampled at 2*dt           
                st = time_conversions.gmst(mjd_time) + dt * (86400.0 / SD) * np.arange(nfft / 2 ) / 3600.0
            
            tt = inj_times + (new_gps_sec - gps0) + num_orig_FFT * tfft_orig / 2 #+ time_conversions.tdt2tdb(mjd_time)

            amps, fsss = inj_provider(tt, num_orig_FFT)

            i1 = np.mod(np.floor(st * (nsid - 1) / 24.0 + 0.5).astype(int),nsid - 1)
            sid1 = inj_provider.ctx.sid1[i1]
            sid2 = inj_provider.ctx.sid2[i1]

            if downsamp:
                fsss = fsss - fr1
            
            if np.sum(np.abs(full_sft)) == 0:
                full_sps = full_sft[k1:k2] * 0
                full_sft = full_sps
#             else:
#                 spec_freqs = freqs[k1:k2]
# #                 plt.semilogy(spec_freqs,np.abs(full_sft[k1:k2]))
# #                 plt.show()
            else:

                phase_evol = inject.phase_from_frequency(tt,fsss)
                sig_t = amps * (sour.Hp() * sid1 + sour.Hc() * sid2) * np.exp(1j * phase_evol)
                # sig_t = amps * np.exp(1j * phase_evol) 
                if downsamp:
                    full_sft = inject.inject_sig_into_sft(sig_t,full_sft[k1:k2],NORM)
                else:
                    full_sft = inject.inject_sig_into_sft(sig_t,full_sft,NORM)
                    full_sft = full_sft[k1:k2]

                full_sps = full_sps[k1:k2]
        else:
            if band:
                full_sft = full_sft[k1:k2]
                full_sps = full_sps[k1:k2]
            
        all_FFTs[jj, :] = full_sft
        all_SPSs[jj, :] = full_sps
        if white_noise:
            tf_map[jj, :] = np.abs(full_sft)**2 / (full_sps / norm_factor) **2
        else:
            tf_map[jj, :] = np.abs(full_sft)**2 / full_sps **2

        all_t0s.append(new_gps_sec)
        ind += 1

    return all_t0s,freqs,all_FFTs.T, all_SPSs.T, tf_map.T



def sub_whitenoise(lfft, normd,normw,ampnoise=7.94e-24):
    """
    Generate complex white noise SFT (column vector style).

    Parameters
    ----------
    lfft : int
        FFT length
    ampnoise : float
        Noise amplitude scaling
    header : object
        Must have attributes:
            header.normd
            header.normw

    Returns
    -------
    sft : ndarray, shape (lfft,)
        Complex white noise vector
    """

    scale = (
        (1.0 / normd) *
        (1.0 / normw) *
        ampnoise *
        (1.0 / np.sqrt(2.0))
    )

    sft = (np.random.randn(lfft) + 1j * np.random.randn(lfft)) * scale

    median_abs = np.median(np.abs(sft))

    sps = (median_abs + np.zeros_like(sft,dtype='float')) \
          * normd \
          * normw \
          * np.sqrt(2.0)

    sps = sps * np.sqrt(1.0 / np.log(2.0))

    return sft,sps

import numpy as np


def calc_dsfact(dt, minf,maxf):
    """
    Compute downsampling factor and normalization.

    Parameters
    ----------
    dt : float
        Original time sampling interval.
    minf : array-like
    maxf : array-like
        min and max freqeuncies of the band to downsample to

    Returns
    -------
    dsfact : float
        Downsampling factor: dsfact = dt_new / dt_old
    NORM : float
        FFT normalization factor: NORM = 1/2 * dsfact
    """

    f_band_you_want = maxf - minf
    dt_prime = 1.0 / f_band_you_want
    dsfact = dt_prime / dt

    if dsfact > 1:
        # normalization for FFT because of downsampling
        # factor 2 due to complex vs real convention (PIA 12 Oct 2016)
        NORM = np.sqrt(dsfact * dsfact) / 2.0
        # equivalently: NORM = dsfact / 2.0
    else:
        NORM = 1.0 / 2.0

    return dsfact, NORM


def get_downsampled_times_samps(dtori,nsamples, dsfact, sft):
    """
    Compute new sampling time and FFT length after downsampling.

    Parameters
    ----------
    header : object or dict
        Must contain:
            - tsamplu   (original time sampling)
            - nsamples  (number of stored samples)
    dsfact : float
        Downsampling factor
    sft : array-like
        SFT data array

    Returns
    -------
    dtnew : float
        New sampling time
    lfftnew : int
        New FFT length
    """

    lfftori = nsamples * 2  # times 2 because extends to negative freqs

    if dsfact > 1:
        dtnew = dtori * dsfact
        lfftnew = lfftori / dsfact

        # Ensure lfftnew is integer-consistent with sft length
        if not np.isclose(np.floor(lfftnew), lfftnew):
            lfftnew_int = int(np.floor(lfftnew))
            lala = len(dtnew * np.arange(lfftnew_int))

            if lala < len(sft):
                lfftnew = int(np.ceil(lfftnew))
            elif lala > len(sft):
                lfftnew = int(np.floor(lfftnew))
            else:
                lfftnew = lfftnew_int
        else:
            lfftnew = int(lfftnew)

    else:
        lfftnew = lfftori
        dtnew = dtori

    return dtnew, lfftnew


def get_sft_sps_and_f_inds(minf,maxf, nsamp,dfr, red,dsfact):
    """
    Translate MATLAB function to Python with 0-based indexing.
    
    Parameters:
    -----------
    freq : array-like
        Frequency range [freq_min, freq_max]
    dsfact : int
        Downsampling factor
        
    Returns:
    --------
    k1, k2, fr1, fr2, kss1, kss2, k1int, k2int : int or float
        Various frequency indices and values
    """
#     nsamp = piahead.nsamples
#     dfr = piahead.deltanu
    inifr0 = 0 * dfr
    finfr0 = (nsamp - 1) * dfr + inifr0
    
    # Convert from MATLAB 1-based to Python 0-based indexing
    k1 = int(np.floor(minf / dfr + 0.0001))
    fr1 = k1 * dfr
    
    k2 = int(np.round(maxf / dfr)) - 1  # Subtract 1 for 0-based indexing
    
    if dsfact > 1:
        if k2 > nsamp - 1:  # Adjust for 0-based indexing
            k2 = nsamp - 1
            if np.floor((k2 - k1) / 2) * 2 == k2 - k1:
                k2 = k2 - 1
    
    fr2 = k2 * dfr
    
    frss1 = max(fr1, inifr0)
    kss1 = int(np.round(frss1 / (dfr * red)))
    
    frss2 = min(fr2, finfr0)
    kss2 = int(np.round(frss2 / (dfr * red)))
    
    var1 = kss2 - kss1 + 1
    var2 = k2 - k1 + 1
    num = var2 / var1
    
    if var1 != var2 and np.floor(num) != num:
        kss1 = kss1 + 1
        var1 = kss2 - kss1 + 1
        num = var2 / var1
        if np.floor(num) != num:
            kss1 = kss1 + 1
    
    if dsfact > 1:
        f0int = np.floor(fr1)
        k1int = int((fr1 - f0int) / dfr)  # Adjust for 0-based indexing
        k2int = int(np.round((fr2 - f0int) / dfr))  # Adjust for 0-based indexing
    else:
        k1int = k1
        k2int = k2
    
     # Adjust k2 and k2int for Python's exclusive-end slicing convention
    k2 = k2 + 1
    k2int = k2int + 1
    return k1, k2, fr1, fr2, kss1, kss2, k1int, k2int


def test_exp_dist(xv,bins=200):

    # xv = whole_map[(freqs>minf) & (freqs<maxf),:].flatten()
    plt.hist(
        xv,
        bins=bins,
        density=True,
        histtype='step',
        lw=2.0,
        color='tab:green',
        label='data'
    )

    # Theoretical exponential PDF (mean=1)
    xx = np.linspace(0, np.percentile(xv, 99.9), 2000)  # just for drawing the curve nicely
    plt.plot(
        xx,
        np.exp(-xx),
        'r--',
        lw=2.2,
        label=r'$\exp(-x)$'
    )

    plt.xlabel(r'$|{\rm FFT}|^2 / {\rm PSD}$', fontsize=14)
    plt.ylabel('Probability density', fontsize=14)

    plt.grid(True, which='major', alpha=0.35)
    plt.grid(True, which='minor', alpha=0.15)
    plt.minorticks_on()

    # No tail clipping of the data, but you *do* want sane axes for visibility:
    plt.xlim(0, np.percentile(xv, 99.5))
    plt.ylim(bottom=0)

    plt.legend(frameon=True, fontsize=12, loc='upper right')

    plt.tight_layout()

    print("mean:", np.mean(xv))
    print("std:", np.std(xv))
    print("median:", np.median(xv))