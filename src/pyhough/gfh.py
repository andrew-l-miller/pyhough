import numpy as np

def hfdf_hough_transients(peaks, hm_job):
    Day_inSeconds = 86400

    gridk = np.squeeze(hm_job['gridk'])
    minf0 = hm_job['minf']
    maxf0 = hm_job['maxf']
    df = hm_job['df']
    enh = hm_job['frenh']
    ref_perc_time = hm_job['ref_perc_time']
    braking_index = hm_job['n']
    poww = braking_index - 1

    if braking_index == 5 or braking_index == 3 or braking_index == 7:
        pass
    else:
        print('chirp, flipping spindowns to spinups')
        gridk = -gridk

    epoch = np.percentile(peaks[0, :], ref_perc_time*100)
    peaks[0, :] = peaks[0, :] - epoch

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
        t = peaks[0, ii0] * Day_inSeconds
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

    hm_job['epoch'] = epoch
    hm_job['dx'] = dx
    hm_job['gridx'] = np.arange(inix, finx, ddx)

    return binh_df0, hm_job

def make_hm_job_struct(minf, maxf, TFFT, dur, n, ref_perc_time, gridk):
    hm_job = {
        'minf': minf,               # minimum frequency to do the Hough on
        'maxf': maxf,               # maximum frequency to do the Hough on
        'df': 1 / TFFT,             # step in frequency
        'dur': dur,
        'patch': [0, 0],
        'n': n,
        'ref_perc_time': ref_perc_time,
        'frenh': 1,
        'gridk': gridk
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

