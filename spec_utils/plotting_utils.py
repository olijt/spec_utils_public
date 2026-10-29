import matplotlib.pyplot as plt
import numpy as np

'''how to get color map easily'''
#cmap = plt.cm.Blues
# colors = cmap(np.linspace(0.1, 1.0, len(antidiag_slices)))  # avoid very light near-white

'''Make figures universally nicer'''
# matplotlib.rcParams.update({
#     "font.size": 14,
#     "axes.titlesize": 14,
#     "axes.labelsize": 14,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "legend.fontsize": 14,
#     "figure.autolayout": True  # This enables tight_layout by default
# })

'''Make figures black for MGB presentations'''
# matplotlib.rcParams.update({
#     "figure.facecolor": "black",
#     "axes.facecolor": "black",
#     "axes.edgecolor": "white",
#     "axes.labelcolor": "white",
#     "xtick.color": "white",
#     "ytick.color": "white",
#     "text.color": "white",
#     "legend.facecolor": "black",
#     "legend.edgecolor": "white",
# })
def plot_heatmap(array, color='plasma', x_vals=None, y_vals=None,
                 num_x_ticks=5, num_y_ticks=5, xlabel='', ylabel='', cbar_label=''):
    # Ensure input is a numpy array
    array = np.asarray(array)
    rows, cols = array.shape

    # Default x and y values to range(length) if not provided
    if x_vals is None:
        x_vals = np.arange(cols)
    if y_vals is None:
        y_vals = np.arange(rows)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    im = ax1.imshow(array, cmap=color, aspect='auto')

    # Ensure tick count doesn't exceed array dimensions
    num_x_ticks = min(num_x_ticks, len(x_vals))
    num_y_ticks = min(num_y_ticks, len(y_vals))

    # Calculate tick positions
    x_tick_indices = np.linspace(0, len(x_vals) - 1, num_x_ticks, dtype=int)
    y_tick_indices = np.linspace(0, len(y_vals) - 1, num_y_ticks, dtype=int)

    # Helper to format labels safely (handles strings or numbers)
    def format_label(val):
        return f'{val:.2f}' if isinstance(val, (int, float, np.number)) else str(val)

    # Set x-axis ticks and labels
    ax1.set_xticks(x_tick_indices)
    ax1.set_xticklabels([format_label(x_vals[i]) for i in x_tick_indices], size=12)
    ax1.set_xlabel(xlabel, size=14)

    # Set y-axis ticks and labels
    ax1.set_yticks(y_tick_indices)
    ax1.set_yticklabels([format_label(y_vals[i]) for i in y_tick_indices], size=12)
    ax1.set_ylabel(ylabel, size=14)

    # Invert to place [0,0] at the bottom-left (standard plot style)
    ax1.invert_yaxis()

    # Add Colorbar
    cbar = fig.colorbar(im, ax=ax1)
    if cbar_label:
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", size=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.show()

def plot_peak_fwhm_hist(peaks, fwhms, inhom_x_y=None, n_bins=None, bin_edges=None):

    """
    Plots histogram of peak energies overlaid with boxplot of fwhms within each bin
    Parameters
    ----------
    peaks: 1D array of peaks
    fwhms: 1D array of fwhms (must correspond to peaks)
    inhom_x_y: tuple of inhomogeneous dist. x and y arrays
    n_bins
    bin_edges

    Returns
    -------

    bin_edges
    """
    fig, ax = plt.subplots(ncols=1, figsize=(10, 8))

    # 1. Get the binning info
    if n_bins is not None:
        counts, bin_edges, _ = ax.hist(peaks, bins=n_bins, color='grey', alpha=0.7, zorder=0)
    elif bin_edges is not None:
        counts, bin_edges, _ = ax.hist(peaks, bins=bin_edges, color='grey', alpha=0.7, zorder=0)
    else:
        counts, bin_edges, _ = ax.hist(peaks, color='grey', alpha=0.7, zorder=0)

    # 3. Create a smooth range of X values to plot the curve
    ax.axvline(2740, color='k', ls='--', lw=1, label='Cutoff')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # 2. Group the data
    indices = np.digitize(peaks, bin_edges)

    binned_data = []
    valid_centers = []

    for i in range(1, len(bin_edges)):
        mask = (indices == i)
        if np.any(mask):
            # Store the actual peak values and fwhm values for this bin
            binned_data.append(fwhms[mask])
            # Store the actual X positions from 'peaks' for these points
            valid_centers.append(peaks[mask])

    if inhom_x_y is not None:
        x = inhom_x_y[0]
        y = inhom_x_y[1].copy()
        y *= max(counts)
        ax.plot(x, y, color='k', label='Pop. Dist.')

    ax.set_ylabel('Counts')
    ax.set_xlabel('Energy (meV)')
    ax.set_xlim(2500, 3000)

    # Create the twin axis
    ax0_fwhm = ax.twinx()

    if binned_data:
        multi_data = []
        multi_centers = []
        single_data = []
        single_centers = []

        for data, actual_x in zip(binned_data, valid_centers):
            if len(data) > 1:
                multi_data.append(data)
                # For boxplots, we usually still want to center them on the bin
                # But if you want them centered on the mean of the data, use np.mean(actual_x)
                bin_idx = np.digitize(actual_x[0], bin_edges) - 1
                multi_centers.append(bin_centers[bin_idx])
            else:
                # Use the actual energy value for the single point
                single_data.append(data[0])
                single_centers.append(actual_x[0])  # <--- This is the precise X value

        # 1. Plot boxes
        if multi_data:
            ax0_fwhm.boxplot(multi_data,
                             positions=multi_centers,
                             widths=bin_width * 0.6,
                             manage_ticks=False,
                             patch_artist=True,
                             boxprops=dict(facecolor='red', alpha=0.3, color='red'),
                             medianprops=dict(color='black', linewidth=2))

        # 2. Plot circles (now at exact energy locations)
        if single_data:
            ax0_fwhm.scatter(single_centers, single_data,
                             color='red', edgecolors='black',
                             s=50, zorder=3, label='Single Point')

    # -------------------------------------------------------------
    # CRITICAL FIX HERE: Force the limits *after* boxplot executes
    # -------------------------------------------------------------
    ax0_fwhm.set_xlim(ax.get_xlim())

    ax0_fwhm.set_ylabel('FWHM (meV)', color='red')
    ax0_fwhm.set_ylim(35, 220)
    ax0_fwhm.tick_params(axis='y', labelcolor='red')

    plt.show()