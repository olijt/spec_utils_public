import numpy as np
from .math_utils import wavelength_to_ev
from scipy.interpolate import UnivariateSpline
'''Dealing with arrays'''
def get_index(array, value):
    return np.argmin(np.abs(array-value))

def get_index_for_timestamps(array, value):
    """

    Parameters
    ----------
    array: MUST BE SORTED
    value

    Returns
    -------

    """

    idx = np.searchsorted(array, value)

    if idx == 0:
        return 0

    if idx == len(array):
        return len(array) - 1

    before = idx - 1
    after = idx

    if value - array[before] < array[after] - value:
        return before
    else:
        return after
def find_fwhm(energy, spectrum):
    '''

    Parameters
    ----------
    energy
    spectrum

    Returns
    -------
    fwhm, center_energy, left_idx, right_idx
    '''
    # Find the peak of the fit and its corresponding center
    max_intensity = max(spectrum)
    center_index = np.argmax(spectrum)
    center_energy = energy[center_index]

    half_max = max_intensity / 2

    # Find the points where the fit crosses half of the maximum intensity
    left_idx = np.where(spectrum >= half_max)[0][0]  # First crossing
    right_idx = np.where(spectrum >= half_max)[0][-1]  # Last crossing

    # FWHM is the distance between these two points
    fwhm = np.abs(energy[right_idx] - energy[left_idx])

    return fwhm, center_energy, left_idx, right_idx


def get_fwhm_spline(x, y):
    """Calculate FWHM of a peak using spline interpolation."""
    # Assuming baseline is near 0. If there's a heavy background,
    # you might want: half_max = (np.max(y) - np.min(y)) / 2 + np.min(y)
    half_max = np.max(y) / 2.0

    # Spline shifted by half_max so roots are the crossing points
    spline = UnivariateSpline(x, y - half_max, s=0)
    roots = spline.roots()

    if len(roots) >= 2:
        # Distance between the first and last root
        return abs(roots[-1] - roots[0])
    return None
def bounds_for_truncating_axis(y, central_index, threshold):

    """
    Search outward from center to find point where function crosses threshold.
    Assumes peaked function

    Parameters
    ----------
    y
    central_index
    threshold: absolute

    Returns
    -------
    left_index, right_index
    """
    ''''''


    # Search to the left
    left_index = central_index
    while left_index > 0 and y[left_index] >= threshold:
        left_index -= 1

    # Search to the right
    right_index = central_index
    while right_index < len(y) - 1 and y[right_index] >= threshold:
        right_index += 1

    return left_index, right_index
def truncate_all_arrays(list_of_arrays, left_index, right_index):
    return [truncate_array(arr, left_index, right_index) for arr in list_of_arrays]
def truncate_array(array, left_index, right_index):
    array = np.asarray(array)  # ensure it's a numpy array
    if array.ndim == 1:
        return array[left_index:right_index]
    elif array.ndim == 2:
        return array[:, left_index:right_index]
    else:
        raise ValueError("Only 1D or 2D arrays are supported.")
def roll_with_zeros(arr, shift):
    '''Rolls an array by some shift amount. Rolling over the edge of the x axis gives zero on the other side'''
    rolled = np.roll(arr, shift)  # Use np.roll to shift the array
    if shift > 0:
        rolled[:shift] = 0  # Set the first 'shift' elements to zero if rolling forward
    elif shift < 0:
        rolled[shift:] = 0  # Set the last 'abs(shift)' elements to zero if rolling backward
    return rolled
def remove_outliers(array):
    '''

    Parameters
    ----------
    array

    Returns
    -------

    '''
    # Make a copy of the array to avoid modifying the original data
    cleaned_array = np.copy(array)

    # Loop through each intensity value except the first and last to avoid index errors
    for i in range(1, len(cleaned_array) - 1):
        # Calculate the average of the neighboring points
        left_neighbor = cleaned_array[i -1]
        right_neighbor=cleaned_array[i+1]


        # Check if the current point is an outlier
        if cleaned_array[i] > max(array)/5 + min(left_neighbor,right_neighbor):
            # Replace the outlier with the averag# intensities = remove_outliers(intensities)e of its neighbors
            cleaned_array[i] = min(left_neighbor,right_neighbor)

    return cleaned_array
def wavelength_spectrum_to_energy_spectrum(wavelength, spectrum):
    """
    For converting emission spectrum between wavelength and energy axes.

    Parameters
    ----------
    wavelength: in nm
    spectrum

    Returns
    -------
    energy, energy_spectrum: energy in meV
    """
    energy = wavelength_to_ev(wavelength)[::-1]
    energy_spectrum = spectrum * (wavelength ** 2) / (1239.84*1000) #scale by Jacobian to keep bins scaled correctly
    energy*=1000 #convert to meV
    energy_spectrum = energy_spectrum[::-1]
    return energy, energy_spectrum

def wavelength_2d_spectrum_to_energy_2d_spectrum(wavelength, spectrum_2d):

    if spectrum_2d.shape[0] != len(wavelength) or spectrum_2d.shape[1] != len(wavelength):
        raise ValueError("Both dimensions of spectrum_2d must match the length of the wavelength axis.")

    energy_axis = wavelength_to_ev(wavelength)[::-1] * 1000  # in meV

    # The 1D Jacobian factor scales density from per-nm to per-meV.
    jacobian_1d = (wavelength ** 2) / (1239.84 * 1000)

    # The 2D Jacobian is the outer product of the 1D Jacobians for each axis.
    jacobian_2d = np.outer(jacobian_1d, jacobian_1d)

    # scale the original spectrum by the Jacobian.
    energy_spectrum_2d = spectrum_2d * jacobian_2d

    # Because we reversed the axes (low energy to high energy), we must also
    # flip the data matrix on both dimensions so the data points correspond
    # to their new axis locations.
    energy_spectrum_2d = energy_spectrum_2d[::-1, ::-1]

    return energy_axis, energy_spectrum_2d
def wavelength_filter_to_energy_filter(wavelength, filter_spectrum):
    """
    Transforms a filter transmission curve from wavelength to energy domain
    without altering transmission percentages. Avoids distortion associated with Jacobian
    """
    energy = wavelength_to_ev(wavelength)[::-1] * 1000  # meV
    filter_energy_spectrum = filter_spectrum[::-1]       # Just reverse order
    return energy, filter_energy_spectrum
def wav_fwhm_to_energy_fwhm(wav_cent, wav_fwhm):
    left = wav_cent - wav_fwhm / 2 #lower wave
    right = wav_cent + wav_fwhm / 2 #higher wave

    left_energy = wavelength_to_ev(left)*1000 #higher energy
    right_energy = wavelength_to_ev(right)*1000 #lower energy

    fwhm = left_energy-right_energy

    return fwhm
def extend_bin_edges(bin_edges, data):

    """
    Useful for making histograms with the same binedges for sets of data that might have different mins and maxes
    Parameters
    ----------
    bin_edges
    data

    Returns
    -------

    new_bin_edges

    """
    data_min=np.min(data)
    data_max=np.max(data)
    edge_spacing = bin_edges[1] - bin_edges[0]

    # Calculate how many bins are needed to cover the data outliers
    n_low_bins_to_add = int(np.ceil(max(0, (bin_edges[0] - data_min) / edge_spacing)))
    n_high_bins_to_add = int(np.ceil(max(0, (data_max - bin_edges[-1]) / edge_spacing)))

    # Generate the new lower edges
    # We go from (bin_edges[0] - n * spacing) up to bin_edges[0]
    low_edges = bin_edges[0] - np.arange(n_low_bins_to_add, 0, -1) * edge_spacing

    # Generate the new upper edges
    # We go from (bin_edges[-1] + spacing) up to (bin_edges[-1] + n * spacing)
    high_edges = bin_edges[-1] + np.arange(1, n_high_bins_to_add + 1) * edge_spacing

    # Combine everything
    new_bin_edges = np.concatenate([low_edges, bin_edges, high_edges])

    return new_bin_edges
def flatten_list(nested_list):
    '''Only for 2D lists or lists of 1D arrays'''
    new_list = []
    for entry in nested_list:
        # Check if the entry is a list or a numpy array
        if isinstance(entry, (list, np.ndarray)):
            for subentry in entry:
                new_list.append(subentry)
        else:
            # This handles cases where the list isn't perfectly 2D
            # (e.g., [1, [2, 3], 4])
            new_list.append(entry)
    return new_list

def get_antidiagonal(array):
    return np.fliplr(array).diagonal()


def get_antidiagonal_slices(array, filter_centers):
    """
    Returns
    -------
    slices: list of tuples (x_physical_meV, y_values)
    """
    n = array.shape[0]
    slices = []

    e1_centers = filter_centers[0]
    e2_centers = filter_centers[1]

    for k in range(2 * n - 1):
        antidiag = []
        x_phys = []
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                antidiag.append(array[i, j])

                # Project onto the anti-diagonal axis for physical distance
                # E1 goes up, E2 goes down, making this strictly increasing
                coord_meV = (e1_centers[i] - e2_centers[j]) / np.sqrt(2)
                x_phys.append(coord_meV)

        slices.append((np.array(x_phys), np.array(antidiag)))

    return slices


def get_diagonal_slices(array:np.array):
    """
    Parameters
    ----------
    array : 2D array

    Returns
    -------
    slices : list of tuples
        Each entry is (x, diag), where:
        - x is the centered coordinate along the diagonal
        - diag contains the diagonal values
    """

    n = array.shape[0]
    slices = []

    # diagonals indexed by k = j - i
    for k in range(-(n - 1), n):
        diag = []
        x = []

        for i in range(n):
            j = i + k

            if 0 <= j < n:
                diag.append(array[i, j])
                x.append(i + j)  # coordinate along the diagonal

        slices.append((np.array(x), np.array(diag)))

    return slices

def coarsen_array_mean(array, factor, return_full=True):
    array = np.asarray(array)

    end = array.shape[0] - (array.shape[0] % factor)

    out = array[:end].reshape(
        end // factor,
        factor,
        *array.shape[1:]
    ).mean(axis=1)

    if return_full:
        if array.shape[0] % factor:
            out = np.concatenate([out, array[end:]], axis=0)

    return out
def coarsen_array_sum(array, factors):
    """
    Sum blocks of size given by factors along each dimension.

    Parameters
    ----------
    array : ndarray
    factors : int or tuple
        Coarsening factor for each axis.
        If int, only coarsens axis 0.

    Returns
    -------
    ndarray
    """
    array = np.asarray(array)

    if np.isscalar(factors):
        factors = (factors,)

    factors = tuple(factors)

    if len(factors) > array.ndim:
        raise ValueError("More factors than array dimensions")

    out = array

    for axis, factor in enumerate(factors):
        if factor == 1:
            continue

        size = out.shape[axis]
        end = size - (size % factor)

        # Keep only full blocks
        slices = [slice(None)] * out.ndim
        slices[axis] = slice(0, end)
        trimmed = out[tuple(slices)]

        # Reshape to split the axis into (n_blocks, factor)
        new_shape = (
            *trimmed.shape[:axis],
            end // factor,
            factor,
            *trimmed.shape[axis + 1 :]
        )

        trimmed = trimmed.reshape(new_shape)

        # Sum over factor dimension
        trimmed = trimmed.sum(axis=axis + 1)

        # Append leftovers unchanged
        if size % factor:
            slices[axis] = slice(end, None)
            remainder = out[tuple(slices)]
            out = np.concatenate([trimmed, remainder], axis=axis)
        else:
            out = trimmed

    return out
def bin_edges_to_bin_centers(bin_edges, mode='linear'):
    if mode=='linear':
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    elif mode=='geometric':
        bin_centers = np.sqrt(bin_edges[:-1]* bin_edges[1:])
    else:
        print('Invalid mode')
        raise ValueError
    return bin_centers
def linear_log_bin_edges(start_value, max_bin_edge, lin_bin_increment, num_lin_bins, later_bin_prec=3,
                         later_bin_scaling_factor=2):
    '''
    Provides mix of linear and log spaced bins. Useful for data which needs resolution at short time, then becomes log spaced
    Parameters
    ----------
    first_bin_edge
    max_bin_edge
    num_lin_bins
    later_bin_coarseness
    later_bin_scaling_factor

    Returns
    -------

    '''

    linear_bin_edges = start_value+lin_bin_increment * np.arange(num_lin_bins + 1)

    later_bin_edges = np.zeros(100_000)
    integer_factor = 1
    for i in range(len(later_bin_edges)):
        if i == 0:
            later_bin_edges[i] = linear_bin_edges[-1]
        else:
            later_bin_edges[i] = later_bin_edges[i - 1] + integer_factor * lin_bin_increment
            if (i + 1) % int(later_bin_prec) == 0:
                integer_factor *= int(later_bin_scaling_factor)
            if later_bin_edges[i] > max_bin_edge:
                break
    later_bin_edges_filtered = later_bin_edges[later_bin_edges > 0]  # remove excess bins
    later_bin_edges_filtered = later_bin_edges_filtered[1:]  # The first nonlinear edge is redundant

    bin_edges = np.hstack((linear_bin_edges, later_bin_edges_filtered))
    bin_centers = bin_edges_to_bin_centers(bin_edges, mode='linear')

    return bin_edges
def bin_square_array(bin_centers, array, new_bin_edges):
    num_new_bins = len(new_bin_edges) - 1
    new_bin_centers = np.zeros(num_new_bins)

    num_axes=len(array.shape)

    if num_axes == 2:
        array_binned = np.zeros((num_new_bins, num_new_bins))

        for i in range(num_new_bins):
            # Mask for the first axis (rows)
            mask_i = (bin_centers >= new_bin_edges[i]) & (bin_centers < new_bin_edges[i + 1])
            n_i = np.sum(mask_i)  # Number of rows in this bin

            if n_i > 0:
                new_bin_centers[i] = np.mean(bin_centers[mask_i])
            else:
                new_bin_centers[i] = (new_bin_edges[i] + new_bin_edges[i + 1]) / 2

            for j in range(num_new_bins):
                # Mask for the t2 axis (columns)
                mask_j = (bin_centers >= new_bin_edges[j]) & (bin_centers < new_bin_edges[j + 1])
                n_j = np.sum(mask_j)  # Number of columns in this bin

                if n_i > 0 and n_j > 0:
                    # Divide the sum by the area (n_i * n_j) to get the average intensity, not the total
                    array_binned[i, j] = np.sum(array[np.ix_(mask_i, mask_j)]) / (n_i * n_j)
    elif num_axes == 1:
        array_binned = np.zeros(num_new_bins)
        for i in range(num_new_bins):
            mask_i = ((bin_centers >= new_bin_edges[i]) & (bin_centers < new_bin_edges[i + 1]))
            n_i = np.sum(mask_i)

            # New bin center
            if n_i > 0:
                new_bin_centers[i] = np.mean(bin_centers[mask_i])
                # Average values in this bin
                array_binned[i] = np.mean(array[mask_i])
            else:
                # No data in this bin, leave it with zero
                new_bin_centers[i] = (new_bin_edges[i] + new_bin_edges[i + 1]) / 2
    else:
        print('Invalid dimensions')
        raise ValueError

    return new_bin_centers, array_binned

def moving_average(x, window):
    kernel = np.ones(window)

    sums = np.convolve(x, kernel, mode="same")
    counts = np.convolve(np.ones_like(x), kernel, mode="same")

    return sums / counts

import types
__all__ = [name for name, obj in globals().items()
           if not name.startswith('_') and isinstance(obj, types.FunctionType)]
