import numpy as np
import pyfstat
import matplotlib.pyplot as plt

def get_detector_velocities(data):
    states = pyfstat.DetectorStates().get_multi_detector_states_from_sfts(data.sftfilepath, central_frequency=data.F0, time_offset=0)
    ts = np.array([data.tGPS.gpsSeconds for data in states.data[0].data])
    velocities = np.vstack([data.vDetector for data in states.data[0].data]).T
    return ts, velocities
    
    
def astro2rect(a, icrad=0):
    # Conversion from astronomical to rectangular coordinates
    # position = [signal_parameters["Alpha"], signal_parameters["Delta"]]
    # r = astro2rect(position,1) #icrad = 1: inputs are in rads; 0: inputs degs

    if len(a) == 2:
        a = np.append(a, 1)

    if icrad == 0:
        deg2rad = np.pi / 180
        a[0] = a[0] * deg2rad
        a[1] = a[1] * deg2rad

    r = np.zeros(3)
    r[0] = np.cos(a[0]) * np.cos(a[1]) * a[2]
    r[1] = np.sin(a[0]) * np.cos(a[1]) * a[2]
    r[2] = np.sin(a[1]) * a[2]

    return r
    
    
def remove_doppler(freqs,vec_n,velocities):

    freqs_dopp_corr = freqs / (1 + np.dot(vec_n, velocities))
    
    return freqs_dopp_corr

def python_plot_triplets(x, y, z, marker, label='', flag_logx=0, flag_logy=0, flag_log_cb=0, colorm='inferno', size=10, show_colorbar=True):
    """
    Plot triplets (x, y, z) with colormap encoding z values.

    Parameters:
    - x, y, z: Data arrays
    - marker: Marker style for plotting
    - label: Label for the colorbar
    - flag_logx, flag_logy: Flags to apply log scale to x or y axes
    - flag_log_cb: Flag to apply log scaling to colorbar values
    - colorm: Colormap to use
    - size: Size of the markers
    - show_colorbar: Whether to display the colorbar (True/False)
    """
    if flag_log_cb == 1:
        z = np.log10(z)

    # Define colormap and normalization
    cmap = plt.get_cmap(colorm)
    norm = plt.Normalize(vmin=np.min(z), vmax=np.max(z))

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Scatter plot with color mapping
    sc = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, marker=marker, s=size)

    # Handle log scales
    if flag_logx:
        ax.set_xscale('log')
    if flag_logy:
        ax.set_yscale('log')

    # Add colorbar if requested
    if show_colorbar:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label)

    return fig, ax


def make_peakmap_from_spectrogram(alltimes,freqs,normalized_power,threshold=3):
    Ntts = len(alltimes)
    freqs_new = []
    times_new = []
    powss_new = []
    index = [0]
    num_peaks = 0
    for t in range(Ntts):
        inds_above_thr = np.array(select_local_max(normalized_power[:,t],threshold))
        num_peaks = num_peaks + len(inds_above_thr)
        index.extend([num_peaks])
        if inds_above_thr.shape[0] == 0:
            continue

        freqs_peaks = freqs[inds_above_thr]        
        freqs_new.extend(freqs_peaks)
        times_new.extend(alltimes[t] * np.ones((len(freqs_peaks),1)))
        powss_new.extend(normalized_power[inds_above_thr,t])
        

    times_new = np.squeeze(times_new)
    freqs_new = np.squeeze(freqs_new)
    powss_new = np.squeeze(powss_new)
    
    return times_new,freqs_new,powss_new,index

def remove_doppler_from_peakmap(times_new,freqs_new,index,vec_n,vs,Nts):
    freqs_undop = []
    for t in range(Nts):
        if t == Nts:
            freqs_one_time = freqs_new[index[t]:]
        else:
            freqs_one_time = freqs_new[index[t]:index[t+1]]
            
        freqs_nodop = remove_doppler(freqs_one_time,vec_n,vs[:,t])
        freqs_undop.extend(freqs_nodop)
    
    return freqs_undop


def select_local_max(pows,threshold):

    peaks_index = list()
    for eqpow in range(1,len(pows)-1):
        if pows[eqpow]>threshold:
            if pows[eqpow]>pows[eqpow+1]:
                if pows[eqpow]>pows[eqpow-1]:
                    peaks_index.append(eqpow)
    return peaks_index

def remove_doppler_from_spectrogram_and_local_max_thresh(alltimes,freqs,normalized_power,vec_n,vs,threshold=3):
#     alltimes = times["H1"]
#     position = [signal_parameters["Alpha"], signal_parameters["Delta"]]
#     vec_n = astro2rect(position,1)
#     ts,vs = get_detector_velocities(data)
    Ntts = len(alltimes)
    freqs_new = []
    times_new = []
    powss_new = []
    for t in range(Ntts):
        inds_above_thr = np.array(select_local_max(normalized_power[:,t],threshold))
        if inds_above_thr.shape[0] == 0:
            continue
        freqs_peaks = freqs[inds_above_thr]
        freqs_dopp = remove_doppler(freqs_peaks,vec_n,vs[:,t])
        freqs_new.extend(freqs_dopp)
        times_new.extend(alltimes[t] * np.ones((len(freqs_dopp),1)))
        powss_new.extend(normalized_power[inds_above_thr,t])



    times_new = np.squeeze(times_new)
    freqs_new = np.squeeze(freqs_new)
    powss_new = np.squeeze(powss_new)
    
    return times_new,freqs_new,powss_new
    

def flatten_spectrogram(alltimes,freqs,normalized_power):
    Ntts = len(alltimes)
    freqs_flat = []
    times_flat = []
    powss_flat = []
    index = [0]
    num_peaks = 0
    for t in range(Ntts):       
        freqs_flat.extend(freqs)
        times_flat.extend(alltimes[t] * np.ones((len(freqs),1)))
        powss_flat.extend(normalized_power[:,t])
        

    times_flat = np.squeeze(times_flat)
    freqs_flat = np.squeeze(freqs_flat)
    powss_flat = np.squeeze(powss_flat)
    
    return times_flat,freqs_flat,powss_flat

def project_peaks(pm_freqs,pm_freqs_undopp):
    """
    Project a peak map onto the frequency axis.

    Parameters:
    - peaks: 2D array where peaks[1, :] contains the frequencies
    - plot_flag: 1 to plot the projected peak map; 0 to skip plotting

    Returns:
    - fbins: The frequency bins in the peak map
    - counts: The number of peaks per frequency bin in the projected peak map
    """
    # Get the unique frequency bins
    fbins = np.unique(pm_freqs)

    # Histogram the frequencies into the unique bins
    counts, fbins = np.histogram(pm_freqs_undopp, bins=fbins)


    return fbins[:-1], counts

def simulate_spectrogram(Ntimes, Nfreqs, mu=1.0):
    """
    Simulate a fake spectrogram where each (time, freq) bin is
    drawn independently from an exponential distribution.

    Parameters:
    - Ntimes: number of time bins
    - Nfreqs: number of frequency bins
    - mu: mean of exponential distribution

    Returns:
    - spectrogram: 2D array of shape (Ntimes, Nfreqs)
    """
    return np.random.exponential(scale=mu, size=(Ntimes, Nfreqs))
