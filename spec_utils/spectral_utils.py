import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .array_utils import get_index, wavelength_spectrum_to_energy_spectrum, find_fwhm
from .fitting_utils import fit_reference_spectrum


'''Loading files into intensity matrix'''


def row_frame_csv_to_intensity_matrix(file_path):
    """

    Parameters
    ----------
    file_path

    Returns
    -------
    wavelength
    intensity_matrix: numpy array with size = (num_frames, wavelength dim)
    """
    df = pd.read_csv(file_path, header=None, skiprows=1)
    wavelength = df.iloc[0].to_numpy().astype(float)
    intensity_indices = range(1, len(df), 3)
    intensity_matrix = df.iloc[intensity_indices].to_numpy().astype(float)

    return wavelength, intensity_matrix


def HC_ASC_to_intensity_matrix(file_path):
    data = np.loadtxt(file_path)

    wavelength = data[:, 0]
    intensity = data[:, 1]

    wavelength, counts = np.unique(wavelength, return_counts=True)

    num_scans = np.unique(counts)
    if len(num_scans) != 1:
        print('Num scans is a non singleton')
        raise ValueError

    num_scans = num_scans[0]
    intensity_matrix = intensity.reshape(num_scans, -1)

    return wavelength, intensity_matrix

'''Dealing with spectral data'''

def check_all_frames(wavelength, intensity_matrix):
    'Plots each frame from the spectral trace'
    for row in intensity_matrix:
        plt.plot(wavelength, row)
    plt.show()
def remove_cosmic_rays(wavelength, intensity_matrix, threshold, window_size=5, plot=True):
    cleaned_matrix = intensity_matrix.copy()
    rows, cols = cleaned_matrix.shape

    # We use a moving median to define what the 'normal' signal should look like
    for r in range(rows):
        row_data = cleaned_matrix[r, :]

        # Calculate a rolling median
        for i in range(window_size, cols - window_size):
            local_window = row_data[i - window_size: i + window_size + 1]
            local_median = np.median(local_window)

            if (row_data[i] - local_median) > threshold:
                # Replace it with the median
                cleaned_matrix[r, i] = local_median
    if plot:
        check_all_frames(wavelength, cleaned_matrix)
    return cleaned_matrix

def get_all_peak_areas(wavelength, intensity_matrix, left_bound, right_bound, plot=False):
    '''define peak wavelength range'''
    left_index = get_index(wavelength, left_bound)
    right_index = get_index(wavelength, right_bound)

    areas = []
    for row in intensity_matrix:
        integrated_peak = np.sum(row[left_index:right_index])
        areas.append(integrated_peak)

    if plot:
        plt.plot(areas)
        plt.xlabel('Frame Index')
        plt.ylabel('Integrated Intensity (440-460 nm)')
        plt.show()

    return areas


def fit_all_frames(wavelength, intensity_matrix,
                   peak_intensity_threshold,
                   num_gaussians, center_bound, fwhm_start, fwhm_lo, fwhm_hi,
                   plot=False):

    #initialize variables to return
    fit_intensity_matrix = np.zeros_like(intensity_matrix)
    background_subtracted_intensity_matrix=np.zeros_like(intensity_matrix)

    wavelength_peaks = []
    wavelength_fwhms = []
    energy_peaks = []
    energy_fwhms = []

    fit_params=[]


    print('Fitting...')
    #Fit each frame
    for i, row in enumerate(intensity_matrix):
        fit, popt, failed = fit_reference_spectrum(wavelength, row, wavelength, num_gaussians, center_bound, fwhm_start, fwhm_lo,
                                        fwhm_hi)
        peak_intensity = max(fit) #use this value so its consistent with spectra viewed before baseline subtraction

        baseline=popt[-1]

        fit_BS=fit-baseline
        row_BS=row-baseline

        background_subtracted_intensity_matrix[i]=row_BS
        fit_intensity_matrix[i] = fit_BS


        if peak_intensity > peak_intensity_threshold and not failed:
            wavelength_peaks.append(wavelength[np.argmax(fit_BS)])
            wavelength_fwhms.append(find_fwhm(wavelength, fit_BS)[0])

            #also get them in terms of energy
            energy, energy_fit = wavelength_spectrum_to_energy_spectrum(wavelength, fit_BS)
            energy_peak=energy[np.argmax(energy_fit)]
            energy_fwhm=find_fwhm(energy, energy_fit)[0]
            print(energy_fwhm)
            energy_peaks.append(energy_peak)
            energy_fwhms.append(energy_fwhm)

            fit_params.append(popt)

            if plot:
                fig, ax= plt.subplots(ncols=2, sharey=True, figsize=(14,8))
                ax[0].plot(wavelength, row, label='Data')
                ax[0].plot(wavelength, fit, lw=3, color='orange', label='Fit')
                ax[1].plot(wavelength, row_BS, label='Data')
                ax[1].plot(wavelength, fit_BS, lw=3, color='orange', label='Fit')
                ax[0].set_ylabel('Intensity (counts)')
                ax[0].set_xlabel('Wavelength (nm)')
                ax[1].set_xlabel('Wavelength (nm)')
                ax[0].set_title('Original')
                ax[1].set_title('Baseline Subtracted')
                fig.suptitle(f'Frame {i}')
                plt.tight_layout()
                plt.show()
        else:
            wavelength_peaks.append(np.nan)
            wavelength_fwhms.append(np.nan)
            energy_peaks.append(np.nan)
            energy_fwhms.append(np.nan)

    wavelength_peaks = np.array(wavelength_peaks)
    energy_peaks = np.array(energy_peaks)
    wavelength_fwhms = np.array(wavelength_fwhms)
    energy_fwhms = np.array(energy_fwhms)

    intensity_matrices=[fit_intensity_matrix, background_subtracted_intensity_matrix]
    peaks_fwhms=[wavelength_peaks, energy_peaks, wavelength_fwhms, energy_fwhms]


    return intensity_matrices, peaks_fwhms, fit_params

def coarsen_spectral_bins(x, spectrum, coarsen_factor):
    """

    Parameters
    ----------
    x
    spectrum
    coarsen_factor

    Returns
    -------
    coarse_x, coarse_spectrum
    """
    # 1. Ensure the array can be perfectly divided by the factor
    trim_length = (len(spectrum) // coarsen_factor) * coarsen_factor
    spectrum = spectrum[:trim_length]
    x = x[:trim_length]

    # 2. Reshape to (New_Length, Coarsen_Factor)
    # The -1 tells numpy to calculate the first dimension automatically
    reshaped_spectrum = spectrum.reshape(-1, coarsen_factor)
    reshaped_x = x.reshape(-1, coarsen_factor)

    # 3. Take the mean along the second axis (axis=1)
    new_spectrum = reshaped_spectrum.mean(axis=1)
    new_x = reshaped_x.mean(axis=1)

    return new_x, new_spectrum