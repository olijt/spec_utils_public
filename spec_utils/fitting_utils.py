import numpy as np
from numba import jit
import time as timing
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve
from .math_utils import fwhm_to_std
from .array_utils import roll_with_zeros
from .photon_stream_utils import trim_data_for_lifetime, convolve_lifetime_w_irf
def exponential(x, A, k, c):
    return A*np.exp(-x/k)+c
def biexponential(x, A1, k1, A2, k2, c):
    return exponential(x, A1, k1, 0)+exponential(x, A2, k2, 0)+c
def triexponential(x, A1, k1, A2, k2, A3, k3, c):
    return exponential(x, A1, k1, 0)+exponential(x, A2, k2, 0)+exponential(x, A3, k3, 0)+c
def n_exponential(x, *params):
    """
    Universal n-exponential model: sum(Ai * exp(-x/tau_i)) + C

    Parameters
    ----------
    x
    params

    Returns
    -------

    """
    # The last parameter is always the offset C
    c = params[-1]
    # All other parameters come in pairs (Amp, Tau)
    amps_taus = params[:-1]

    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(amps_taus), 2):
        A = amps_taus[i]
        tau = amps_taus[i + 1]
        result += A * np.exp(-x / tau)

    return result + c
def fit_lifetime_n_exp(bin_centers, counts, n=3, plot=True, verbose=True):

    """
    Fits lifetimes with sum of exponentials using iterative algorithm
    Note: Performs better for unnormalized data
    Parameters
    ----------
    x: time bins
    y: counts
    n: num gaussians
    plot: whether to show the result

    Returns
    -------
    sorted_popt: optimized parameters, sorted from least to greatest tau, Nx(amp,tau)+(offset)
    fit: array of fit values
    frac_ints: fractional intensity for each component by integrated area (not amplitude)
    effective_lifetime: weighted average of taus by fractional intensity
    """

    fit_x,y, shift=trim_data_for_lifetime(bin_centers,counts)
    # 1. Initial Guess for n=1
    # We start simple: Max amplitude, mid-range tau, and the tail-end average for offset
    current_popt = [np.max(y), (np.max(fit_x) - np.min(fit_x)) / 5, np.mean(y[-10:])]

    # 2. Iterative Fitting Loop
    for i in range(1, n + 1):
        if verbose:
            print(f"Fitting n={i}...")

        # Prepare bounds for the current number of parameters
        lower_bounds = [0] * len(current_popt)
        upper_bounds = [np.inf] * len(current_popt)
        upper_bounds = [np.inf, fit_x[-1]/4] * i +[np.inf]

        try:
            popt, _ = curve_fit(
                n_exponential, fit_x, y,
                p0=current_popt,
                sigma=np.sqrt(np.abs(y) + 1),
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000
            )
            current_popt = list(popt)
        except Exception as e:
            print(f"Fit failed at n={i}: {e}")
            break

        # If we aren't at the final 'n' yet, prepare the seed for the next iteration
        if i < n:
            # We "split" the largest amplitude component or add a new one
            # Insert a new Amp and Tau before the offset (the last element)
            new_amp = current_popt[0] / 2  # Take half of the first amplitude
            current_popt[0] = new_amp  # Adjust the existing one
            new_tau = current_popt[1] * 2  # Guess a slower decay for the new component

            current_popt.insert(-1, new_amp)
            current_popt.insert(-1, new_tau)

            # current_fit = n_exponential(x, *current_popt)
            # if plot:
            #     plt.figure(figsize=(8, 5))
            #     plt.semilogy(x, y, label='Data')
            #     plt.semilogy(x, current_fit, lw=2, label=f'Current {i}-Exp Fit', color='black', ls='--')
            #     plt.legend()
            #     plt.title(f"Progressive Fit Result (Reached n={i})")
            #     plt.show()

    # 3. Final Results and Plotting
    fit_y= n_exponential(fit_x, *current_popt)


    # Dynamic Printout
    amps = current_popt[:-1:2]
    taus = current_popt[1:-1:2]
    offset = current_popt[-1]
    # 2. Sort by Tau (Increasing)
    # We zip them together, sort by the second element (tau), and unpack
    sorted_pairs = sorted(zip(amps, taus), key=lambda x: x[1])
    sorted_amps, sorted_taus = zip(*sorted_pairs)

    sorted_popt = []
    for amp, tau in sorted_pairs:
        sorted_popt.extend([amp, tau])

    # 4. Append the offset back to the end
    sorted_popt.append(offset)

    # 3. Calculate Fractional Intensity
    # Intensity (Area) of an exponential component is A * tau
    try:
        component_intensities = [np.trapezoid(A * np.exp(-fit_x / tau), fit_x) for A, tau in zip(sorted_amps, sorted_taus)]
    except AttributeError:
        component_intensities = [np.trapz(A * np.exp(-fit_x / tau), fit_x) for A, tau in zip(sorted_amps, sorted_taus)]

    total_intensity = sum(component_intensities)

    if verbose:
        print(f"\n--- Final Results (n={len(sorted_amps)}) ---")
        print(f"{'Comp':<6} | {'Amp':<10} | {'Tau':<10} | {'Intensity %':<12}")
        print("-" * 45)

    frac_ints=[]
    amp_avg_lifetime=0
    intensity_avg_lifetime=0
    for j, (a, t) in enumerate(zip(sorted_amps, sorted_taus)):
        # Calculate percentage of total photons/signal contributed by this component
        frac_int = (component_intensities[j] / total_intensity) * 100
        frac_ints.append(frac_int)

        frac_amp=a/np.sum(sorted_amps)*100

        amp_avg_lifetime+=frac_amp*t
        intensity_avg_lifetime+=(component_intensities[j]/total_intensity)*t
        if verbose:

            print(f"{j + 1:<6} | {frac_amp:<10.2f}% | {t:<10.2f} | {frac_int:<11.1f}%")
    if verbose:
        print("-" * 45)
        print(f"Offset (C): {offset:.2f}")

    if plot:
        plt.figure(figsize=(8, 5))
        plt.semilogy(bin_centers, counts, label='Data')
        plt.semilogy(fit_x+shift, fit_y, lw=2, label=f'Final {i}-Exp Fit', color='black', ls='--')
        plt.legend()
        plt.title(f"Progressive Fit Result (Reached n={i})")
        plt.show()

    return sorted_popt, (fit_x, fit_y), frac_ints, (amp_avg_lifetime, intensity_avg_lifetime)
def fit_lifetime_w_irf(signal_x, signal_y, irf_x, irf_y, p0, bounds=None, plot=True, verbose=True):
    """
    Fits full lifetime data (baseline, rising edge, and decay) by optimizing
    multi-exponential parameters alongside a continuous time-shift parameter.
    """
    dt = signal_x[1] - signal_x[0]
    idx_irf_peak = np.argmax(irf_y)

    # 1. Define the continuous-shifting wrapper function for curve_fit
    def objective_func(x, *params):
        # Extract parameters based on our new structure
        c = params[-1]
        t_shift = params[-2]
        amps_taus = params[:-2]

        # Generate a pristine model time axis starting exactly at 0.
        # We extend it slightly beyond signal_x to ensure convolution doesn't truncate.
        t_max = x[-1] - x[0] + (len(irf_y) * dt)
        t_model = np.arange(0, t_max, dt)

        # Calculate the pure multi-exponential decay starting at t=0
        ideal_decay = np.zeros_like(t_model, dtype=float)
        for i in range(0, len(amps_taus), 2):
            A = amps_taus[i]
            tau = amps_taus[i + 1]
            ideal_decay += A * np.exp(-t_model / tau)

        # Convolve the pure decay with the IRF
        conv_raw = convolve(ideal_decay, irf_y, mode='full') * dt

        # Establish the time axis of the convolution.
        # By definition, the point where the decay begins (t=0) meets the IRF peak
        # occurs precisely at index `idx_irf_peak`.
        t_conv = (np.arange(len(conv_raw)) - idx_irf_peak) * dt

        # Use interpolation to continuously shift the convolved model onto signal_x.
        # Any values falling outside the evaluation window safely default to 0.
        conv_shifted = np.interp(x - t_shift, t_conv, conv_raw, left=0, right=0)

        # Add the constant background baseline
        return conv_shifted + c

    # --- Plot Initial Guess Check ---
    plt.figure(figsize=(7, 4))
    plt.semilogy(signal_x, signal_y, label='Data', alpha=0.7)
    plt.semilogy(signal_x, objective_func(signal_x, *p0), '--', label='Initial Guess')
    plt.title("Initial Guess Alignment Check")
    plt.legend()
    plt.show()

    # 2. Perform the optimization
    if bounds is None:
        bounds = (-np.inf, np.inf)

    try:
        popt, pcov = curve_fit(objective_func, signal_x, signal_y, p0=p0, bounds=bounds)
    except RuntimeError as e:
        print(f"Optimal parameters not found: {e}")
        return None, None

    if verbose:
        print("\n--- Fit Results ---")
        for i in range(0, len(popt)-2, 2):
            print(f"Component {i//2 + 1}: Amplitude = {popt[i]:.2f}, Tau = {popt[i+1]:.4f}")
        print(f"Time Shift (t0): {popt[-2]:.4f}")
        print(f"Baseline (C):    {popt[-1]:.2f}\n")

    # 3. Plotting the final results
    if plot:
        best_fit_curve = objective_func(signal_x, *popt)
        irf_scaled = irf_y / np.max(irf_y) * np.max(signal_y)

        plt.figure(figsize=(8, 6))
        plt.semilogy(irf_x, irf_scaled, color='k', label='IRF (scaled)', alpha=0.2)
        plt.semilogy(signal_x, signal_y, label='Measured Data', alpha=0.8)
        plt.semilogy(signal_x, best_fit_curve, '--', color='purple', label='Best Fit Model', linewidth=2)

        min_y = np.min(signal_y[signal_y > 0])
        plt.ylim(bottom=max(1e-4, min_y / 2))

        plt.title('Full-Curve Lifetime Fit (Convolved with IRF)')
        plt.xlabel('Time')
        plt.ylabel('Intensity (Log)')
        plt.legend()
        plt.show()

    return popt, pcov

def stretched_exp(x, A, tt, b, c):

    return np.log(A*np.exp(-(x / tt) ** b) +c)
def FCS_for_calibrate(x, A, w_0, k):

    D=0.044*(1/1.8) #10^-6 cm^2/s
    D*=100 #um^2/s
    t=x

    diff1=(1+4*D*t/(w_0**2))**(-1/2)
    diff2=(1+4*D*t/((k**2)*(w_0**2)))**(-1)


    diff=A*diff1*diff2

    return 1+diff
def FCS(x, A,td):

    k=4.63 #calibrated for sPCFS setup, comment out to let float
    td=10**td

    diff=A*((1+x/td))**-1*((1+x/(td*k**2)))**(-1/2)

    return 1+diff

def FCS_blink(x, A,td, B, m):

    k=4.63 #calibrated for sPCFS setup, comment out to let float
    td=10**td

    B=10**B

    blink=(1-B*x**(2-m))

    diff=A*((1+x/td))**-1*((1+x/(td*k**2)))**(-1/2)

    return 1+diff*blink
def FCS2(x, A,td, tt, b):

    k=4.63
    td=10**td
    tt=10**tt

    blink=(1+np.exp(-x/tt))**b
    diff=A*((1+x/td))**-1*((1+x/(td*k**2)))**(-1/2)

    return 1+diff*blink
def fit_fcs(tau, data, widebounds=True, plot=True, verbose=True):
    '''

    Parameters
    ----------
    tau
    data
    widebounds
    plot
    verbose

    Returns
    -------
    fit
    '''

    conc_init=np.mean(data[0:10])-1
    if conc_init<0:
        conc_init=0
    if verbose:
        print('Conc init', conc_init)
    init = [conc_init, 10, 1, 1]
    # bounds = ([0,  4, 1, 0],
    #           [4,  12, 1.4, 2])  # for FCS

    if plot:
        plt.plot(tau, data, color='red', label='Data')
        plt.semilogx(tau, FCS(tau, *init), color='black', label='Init')
        plt.legend()
        plt.show()

    #use wide bounds if first fit
    bounds= ([0,  8.5, 6.5, 0],
              [1,  10.5, 20, 1])

    init=np.array(bounds[0])+1e-3

    params, cov = curve_fit(FCS, tau, data, p0=init, bounds=bounds)
    params_std = np.sqrt(np.diag(cov))
    if verbose:
        print("Parameter Estimates and Standard Deviations:")
        for i, (param, std) in enumerate(zip(params, params_std)):
            print(f"Parameter {i + 1}: {param:.4f} ± {std:.4f}")

    A_std=params_std[0]
    fit = FCS(tau, *params)

    if plot:
        plt.plot(tau, data,color='red', label='Data')
        plt.plot(tau, fit, linestyle='--', label='Fit')
        plt.xscale('log')
        plt.show()

    if verbose:
        print(params[0]/2)
        print(fit[0])

    return fit

'''Spectral functions'''
def norm_gaussian(x,center, fwhm):
    '''Returns area normalized Gaussian. TAKES FWHM not STD'''
    std=fwhm_to_std(fwhm)
    return (1 / (np.sqrt(2 * np.pi) * std)) *np.exp(-((x-center)**2)/(2* std**2))
def gaussian(x,center, fwhm):
    '''TAKES FWHM not STD. NOT area normalized'''
    std=fwhm_to_std(fwhm)
    if fwhm<0:
        print('Only positive FWHM valid')
        raise ValueError
    return np.exp(-((x-center)**2)/(2* std**2))
def double_gaussian_spectrum(energy, center1, fwhm1, weight2, center2, fwhm2):
    '''Takes sum of gaussians (not area normalized) and then area normalizes. TAKES FWHM not STD'''
    if weight2<0:
        print('Only positive weights valid')
        raise ValueError
    spectrum=gaussian(energy, center1, fwhm1) + weight2 * gaussian(energy, center2, fwhm2)
    try:
        area = np.trapz(spectrum, energy)
    except AttributeError:
        area = np.trapezoid(spectrum, energy)

    spectrum /= area
    return spectrum
def triple_gaussian_spectrum(energy, center1, fwhm1, weight2, center2, fwhm2,weight3, center3, fwhm3):
    '''Takes sum of gaussians (not area normalized) and then area normalizes. TAKES FWHM not STD'''
    if weight2<0 or weight3<0:
        print('Only positive weights valid')
        raise ValueError
    spectrum=gaussian(energy, center1, fwhm1) + weight2 * gaussian(energy, center2, fwhm2)+weight3*gaussian(energy, center3, fwhm3)
    try:
        area = np.trapz(spectrum, energy)
    except AttributeError:
        area = np.trapezoid(spectrum, energy)

    spectrum /= area
    return spectrum
def quad_gaussian_spectrum(energy, center1, fwhm1, weight2, center2, fwhm2,weight3, center3, fwhm3,weight4, center4, fwhm4):
    '''Takes sum of gaussians (not area normalized) and then area normalizes. TAKES FWHM not STD'''
    if weight2<0 or weight3<0:
        print('Only positive weights valid')
        raise ValueError
    spectrum=(gaussian(energy, center1, fwhm1) +
              weight2 * gaussian(energy, center2, fwhm2)+
              weight3*gaussian(energy, center3, fwhm3)+
              weight4*gaussian(energy, center4, fwhm4))
    area = np.trapezoid(spectrum, energy)
    spectrum /= area
    return spectrum
def skewed_gaussian_spectrum(energy, center, fwhm, skew):
    from scipy.stats import skewnorm
    scale=fwhm_to_std(fwhm)
    spectrum=skewnorm.pdf(energy, skew, loc=center, scale=scale)
    spectrum=spectrum/(np.trapz(spectrum, energy))
    return spectrum
def log_distribution(x, dist_center, std):
    '''Generates log normal distribution. This gives the sd shape which is parametrized by both center and std.
    Don't confuse dist_center with the center position of the distribution
    Notes:
    Since only defined for x>0, turn NaN to zero.
    Also flip over y axis since asymmetry is predominantly to the red.'''
    pdf=(((-x*std*np.sqrt(2*np.pi))**-1)*
         (np.exp(-((np.log(-x)-dist_center)**2)/(2*std**2))))
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

    return np.array(pdf)
def log_dist_rolled(energy, center,dist_center, std):
    '''Takes log normal distribution and shifts to correct center frequency'''
    spectrum=log_distribution(energy,dist_center,std)
    area=np.trapz(spectrum, energy)
    spectrum/=area

    max_index=np.argmin(np.abs(spectrum-max(spectrum)))
    center_index=np.argmin(np.abs(energy-center))
    shift=center_index-max_index
    rolled_spectrum=roll_with_zeros(spectrum,shift)

    return rolled_spectrum
def scaled_gaussian(x, amp, center, fwhm):
    std = fwhm_to_std(fwhm)
    if fwhm < 0:
        print('Only positive FWHM valid')
        raise ValueError
    return amp*np.exp(-((x - center) ** 2) / (2 * std ** 2))

def centered_gaussian_spectrum(energy, center, sd_info):
    sd_fwhm, sd_weight2 = sd_info
    spectrum = norm_gaussian(energy, center, sd_fwhm)

    return spectrum
def centered_double_gaussian_spectrum(energy, center, sd_info):
    sd_fwhm, sd_weight2, sd_center2, sd_fwhm2 = sd_info
    spectrum = double_gaussian_spectrum(energy, center, sd_fwhm,
                                        sd_weight2, sd_center2 + center, sd_fwhm2)
    return spectrum

def centered_triple_gaussian_spectrum(energy, center, sd_info):
    sd_fwhm, sd_weight2, sd_center2, sd_fwhm2, sd_weight3, sd_center3, sd_fwhm3 = sd_info
    spectrum = triple_gaussian_spectrum(energy, center, sd_fwhm,
                                        sd_weight2, sd_center2 + center, sd_fwhm2,
                                        sd_weight3, sd_center3 + center, sd_fwhm3)
    return spectrum


def n_gaussians_with_baseline(x, *params):
    """
    Sum of N Gaussians plus a constant baseline.
    Expects params = [A1, mu1, sigma1, A2, mu2, sigma2, ..., AN, muN, sigmaN, baseline]
    Written by JH"""
    n = (len(params) - 1) // 3  # number of Gaussians
    baseline = params[-1]
    result = np.zeros_like(x)
    for i in range(n):
        A = params[3 * i]
        mu = params[3 * i + 1]
        sigma = params[3 * i + 2]
        result += A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return result + baseline
def n_gaussians_without_baseline(x, *params):
    """
    Sum of N Gaussians plus a constant baseline.
    Expects params = [A1, mu1, sigma1, A2, mu2, sigma2, ..., AN, muN, sigmaN]
    Written by JH"""
    n = (len(params)) // 3  # number of Gaussians
    result = np.zeros_like(x)
    for i in range(n):
        A = params[3 * i]
        mu = params[3 * i + 1]
        sigma = params[3 * i + 2]
        result += A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return result
def n_gaussians_with_baseline_OT(x, *params):
    'Modified from JH, take fwhm instead of std to be compatible for OT functs'
    n = (len(params)) // 3  # number of Gaussians
    baseline = params[-1]
    result = np.zeros_like(x, dtype=float)
    for i in range(n):
        A = params[3 * i]
        mu = params[3 * i + 1]
        fwhm = params[3 * i + 2]
        result += scaled_gaussian(x, A, mu, fwhm)
    return result+baseline
def n_gaussians_without_baseline_OT(x, *params):
    'Modified from JH, take fwhm instead of std to be compatible for OT functs'
    n = (len(params)) // 3  # number of Gaussians
    result = np.zeros_like(x)
    for i in range(n):
        A = params[3 * i]
        mu = params[3 * i + 1]
        fwhm = params[3 * i + 2]
        sigma=fwhm_to_std(fwhm)
        result += scaled_gaussian(x, A, mu, fwhm)
    return result
def fit_reference_spectrum(x_ref, y_ref, x_fit, num_gaussians, center_bound, fwhm_start, fwhm_lo, fwhm_hi, plot_init=False, verbose=False):
    ref_center_mu = x_ref[y_ref.argmax()]  # Initialize guesses for center wav.

    initial_guess = [np.max(y_ref) / num_gaussians, ref_center_mu, fwhm_start]
    lower_bounds = [0.01 * np.max(y_ref), ref_center_mu - center_bound, fwhm_lo]
    upper_bounds = [1.3 * np.max(y_ref), ref_center_mu + center_bound, fwhm_hi]

    new_initial_guess, new_lower_bounds, new_upper_bounds = [], [], []

    for i in range(num_gaussians):
        new_initial_guess.extend(initial_guess)
        new_lower_bounds.extend(lower_bounds)
        new_upper_bounds.extend(upper_bounds)

    new_initial_guess.append(0.001)
    new_lower_bounds.append(-1000)
    new_upper_bounds.append(10000000)

    if plot_init:
        plt.plot(x_ref, y_ref)
        plt.plot(x_fit, n_gaussians_with_baseline_OT(x_fit, *new_initial_guess))
        plt.title('Initial Guess')
        plt.show()
    try:
        popt, pcov = curve_fit(n_gaussians_with_baseline_OT, x_ref, y_ref, p0=new_initial_guess,
                           bounds=(new_lower_bounds, new_upper_bounds), ftol=1e-8, maxfev=10000)
        failed=False
    except RuntimeError:
        print("FIT FAILED! Using initial guess for now...")
        popt=new_initial_guess
        popt[0]=0
        popt[4]=0
        failed=True
    baseline = popt[-1]

    if verbose:
        print(popt)

    y_fit_ref = n_gaussians_with_baseline_OT(x_fit, *popt)

    return y_fit_ref, popt, failed


def fit_data_linear(x, y, plot=True, x_label='x data', y_label='y data', verbose=False):

    """

    Parameters
    ----------
    x
    y
    plot
    x_label
    y_label
    verbose

    Returns
    -------
    fit_line
    slope
    intercept
    """
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # 2. Calculate R-squared
    r_squared = r_value ** 2

    # 3. Create the fit line values
    # y = mx + c
    fit_line = slope * np.array(x) + intercept

    # 4. Plotting
    if plot:
        plt.scatter(x, y, label='Data', alpha=0.7)
        plt.plot(x, fit_line, color='red',
                 label=f'Linear Fit ($R^2={r_squared:.3f}$)')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    if verbose:
        print(f"Regression Equation: y = {slope:.4f}x + {intercept:.4f}")
        print(f"R-squared: {r_squared:.4f}")

    return fit_line, slope, intercept

def power_law_exp(x, A, B, C):
    return A*x**-C*np.exp(-B*x)

def stretched_exp_fcs(x, A, tt, b):
    return (1 + A*np.exp(-x / tt)) ** b

def blinking_Orrit(x, A, B, m):
    return A*(1-B*x**(2-m))

import types
__all__ = [name for name, obj in globals().items()
           if not name.startswith('_') and isinstance(obj, types.FunctionType)]