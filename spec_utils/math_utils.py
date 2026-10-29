import numpy as np

'''Pure arithmetic'''
def fwhm_to_std(fwhm):
    return fwhm/(2*np.sqrt(2*np.log(2)))
def wavelength_to_ev(wavelength):
    return 1239.8419/wavelength
def rmse(residuals):
    return np.sqrt(np.mean(residuals**2))
def confidence_intervals(data, level):
    """

    Parameters
    ----------
    data: array of data
    level: % confidence (i.e. 95% confidence interval)

    Returns
    -------

    confidence interval: [lb, ub]

    """
    data = np.asarray(data)

    lower = (1 - level) / 2 * 100
    upper = (1 + level) / 2 * 100
    ci = np.percentile(data, [lower, upper])

    return ci

def wavelength_to_joules(wavelength):

    """

    Parameters
    ----------
    wavelength (in nm)

    Returns
    -------
    energy per photon (joules)
    """

    wavelength*=10**-9 #convert to m
    h = 6.626e-34  # planck
    c = 2.998e8  # m/s

    energy_per_photon = h * c / wavelength #in joules

    return energy_per_photon

def normal_dist_cutoff_JH(n_data, center, sigma, bound):
    sigmas = np.zeros(n_data)
    idxs = np.arange(n_data)
    idxs_to_fill = np.arange(n_data)
    while True:
        new_sigmas = np.random.normal(loc=center, scale=sigma, size=n_data)
        in_idxs = idxs[(new_sigmas > center-bound) & (new_sigmas < center+bound)]
        for in_idx in in_idxs:
            if in_idx in idxs_to_fill:
                sigmas[in_idx] = new_sigmas[in_idx]
                idxs_to_fill[in_idx] = -1

        nonnegs = idxs_to_fill[idxs_to_fill>-1]
        if len(nonnegs) == 0:
            break

    return sigmas
def calculate_exc_rate(wavelength, abs_cross_section, power, na, diff_limited=True, spot_size_area=0):

    """

    Parameters
    ----------
    wavelength (in nm)
    abs_cross_section (in cm^2)
    power (in W)
    na
    diff_limited
    spot_size_area

    Returns
    -------
    exc_rate (abs/s)
    """

    if diff_limited:
        spot_size_diameter = 1.22 * wavelength / na
        spot_size_diameter_cm=spot_size_diameter*10**-7

        area = np.pi * ((spot_size_diameter_cm / 2) ** 2)
    else:
        area=spot_size_area

    avg_intensity=power/area #W/cm^2
    max_intensity=2*avg_intensity #at center of focal spot for Gaussian PSF

    energy_per_photon=wavelength_to_joules(wavelength)
    print(f"Avg intensity (W/cm^2): {avg_intensity:.3f}")

    avg_photon_flux= avg_intensity/energy_per_photon
    print(f"Photons/sec/cm^2 at center: {avg_photon_flux:.3f}")

    avg_abs_per_s= avg_photon_flux*abs_cross_section #abs per second
    print(f"Exc rate: {avg_abs_per_s:.3f} photons absorbed/s")

    print(f"Intensity at center (W/cm^2) : {max_intensity:.3f}")

    peak_photon_flux= max_intensity/energy_per_photon
    print(f"Photons/sec/cm^2 at center: {peak_photon_flux:.3f}")

    max_abs_per_s= peak_photon_flux*abs_cross_section #abs per second
    print(f"Exc rate: {max_abs_per_s:.3f} photons absorbed/s")

    return max_abs_per_s,avg_abs_per_s

def time_domain_to_freq_domain(time, signal):
    dt = np.mean(np.diff(time))
    N = len(signal)

    # Compute real-valued FFT and positive frequency bins
    frequencies = np.fft.rfftfreq(N, d=dt)
    fft_magnitude = np.abs(np.fft.rfft(signal))

    return frequencies, fft_magnitude


import types
__all__ = [name for name, obj in globals().items()
           if not name.startswith('_') and isinstance(obj, types.FunctionType)]
