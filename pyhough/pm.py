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

def python_plot_triplets(x,y,z,marker,label='',flag_logx=0,flag_logy=0,flag_log_cb=0,colorm='inferno'):

    if flag_log_cb==1:
        z=np.log10(z);
    
    #cmap=plt.get_cmap('inferno')
    cmap=plt.get_cmap(colorm)
    nc=cmap.N
    mi=np.min(z);
    ma=np.max(z);
    mami=ma-mi;
    zz=np.floor(nc*0.9999*(z-mi)/mami+1);

    fig,ax=plt.subplots()

    for i in range(nc):
        col=cmap(i)
#         if flag_logx==1 & flag_logy!=1:
#             plt.semilogx(x[zz==i],y[zz==i],marker,markerfacecolor=col,markeredgecolor=col)
#         elif flag_logy==1 & flag_logx!=1:
#             plt.semilogy(x[zz==i],y[zz==i],marker,markerfacecolor=col,markeredgecolor=col)
#         elif flag_logy==1 &flag_logx==1:
#             plt.loglog(x[zz==i],y[zz==i],marker,markerfacecolor=col,markeredgecolor=col)
#         else:
        #cmaplist = [cmap(i) for i in range(cmap.N)]
        plt.plot(x[zz==i],y[zz==i],marker,markerfacecolor=col,markeredgecolor=col)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=mi, vmax=ma))
    plt.colorbar(sm,label=label)
    #plt.clim(mi, ma)
    plt.grid(True)

    return fig,ax

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
