import numpy as np
from pyhough import pm
from pyhough import hm

# 1 month in chunks of 1024 seconds

sec_per_month = 30*86400;
TFFT = 1800

fmax = 101
fmin = 100
df = 1/TFFT

Nfreqs = (fmax - fmin / df)

times = np.arange(0,sec_per_month,TFFT)
freqs = np.arange(fmin,fmax,df)

Ntimes = len(times)
Nfreqs = len(freqs)

spec = pm.simulate_spectrogram(Ntimes, Nfreqs)

pm_times,pm_freqs,pm_pows,index = pm.make_peakmap_from_spectrogram(times,freqs,spec)

assert len(pm_times) == len(pm_freqs)
assert len(pm_freqs) == len(pm_pows)


dsd = df/sec_per_month

sdgrid = hm.make_sd_grid(0.,dsd) 

Nsds = len(sdgrid)



Nf0 = int(np.ceil((np.max(pm_freqs) - np.min(pm_freqs)) / df))

hmap = hm.hfdf_hough(pm_times,pm_freqs,TFFT,sdgrid)

assert np.shape(hmap) == (Nsds, Nf0)
assert np.all(pm_freqs >= fmin)
assert np.all(pm_freqs <= fmax)
