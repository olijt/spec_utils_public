import numpy as np
from numba import jit
import time as timing
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from .fitting_utils import *
from .array_utils import get_index
'''Dealing with photon streams. Oliver Tye 2023-'''

#Raw data
def get_exp_data(file_path, max_events=1000000):

    """

    Parameters
    ----------
    file_path
    max_events

    Returns
    -------

    timestamps, channels, channel_names, n_photons, exp_length
    """

    if file_path.endswith('.ttbin'):
        print('Reading .ttbin file')
        photon_data = get_photons(file_path, max_events)
        timestamps, channels = process_photon_data(photon_data)
    elif file_path.endswith('.mat'):
        #used for reading David's mat files. Probably not general
        print('Reading .mat file')

        import h5py

        with h5py.File(file_path, "r") as f:
            print(list(f.keys()))
            macroA, microA = np.array(f["chA"])
            macroB, microB = np.array(f["chB"])
            macroC, microC = np.array(f["chC"])
            macroD, microD = np.array(f["chD"])
        syncA=macroA-microA
        syncB=macroB-microB
        syncC=macroC-microC
        syncD=macroD-microD

        sync=np.concatenate([syncA, syncB,syncC, syncD])

        channels_sync=[1]*len(sync)
        channelsA=[5]*len(macroA)
        channelsB=[6]*len(macroB)
        channelsC=[7]*len(macroC)
        channelsD=[8]*len(macroD)

        timestamps = np.concatenate([
            sync,
            macroA, macroB, macroC, macroD
        ])

        channels = np.concatenate([
            channels_sync,
            channelsA, channelsB, channelsC, channelsD
        ])

        order = np.argsort(timestamps)

        timestamps = timestamps[order]
        channels = channels[order]
    else:
        print('INVALID FILE TYPE!')
        raise ValueError()


    channel_names = set(channels[:5000])  # just search the first 5000 events
    channel_names=list(channel_names)
    n_photons = len(timestamps)
    exp_length = timestamps[-1]-timestamps[0]


    print('Recorded Channels:', channel_names)
    print(f"Number of channels: {len(channel_names)}")
    print(f"Number of photons: {n_photons}")

    print('Exp Time:', f'{exp_length / 1e12} seconds')
    print('Total avg. count rate', f'{n_photons / (exp_length / 1e12) / 1e3} kcps')

    return timestamps, channels, channel_names, n_photons, exp_length
def get_photons(file_path, max_events):

    import TimeTagger

    print('Reading', file_path)


    # To read the data form the temporary file, we create a FileReader object
    if file_path.endswith('.ttbin'):
        filereader = TimeTagger.FileReader(file_path)


        n_events = max_events  # Number of events to read at once
        i = 0


        channels=[]
        timestamps=[]
        overflow_types=[]
        missed_events=[]

        print('Reading .ttbin ... (each GB requires ~200 its)')
        pbar=tqdm()
        while filereader.hasData():
        # getData() does not return timestamps, but an instance of TimeTagStreamBuffer
        # that contains more information than just the timestamp
            data = filereader.getData(n_events=n_events)

            # With the following methods, we can retrieve a numpy array for the particular information:
            channels.append(data.getChannels())         # The channel numbers
            timestamps.append(data.getTimestamps())       # The timestamps in ps
            overflow_types.append(data.getEventTypes())   # TimeTag = 0, Error = 1, OverflowBegin = 2, OverflowEnd = 3, MissedEvents = 4
            missed_events .append(data.getMissedEvents())  # The numbers of missed events in case of overflow

            pbar.update(1)
        pbar.close()
        channels = np.concatenate(channels)
        timestamps = np.concatenate(timestamps)
        overflow_types = np.concatenate(overflow_types)
        missed_events = np.concatenate(missed_events)

        time_tag_events = (overflow_types == 0)
        channels = channels[time_tag_events]
        timestamps = timestamps[time_tag_events]

        print("Photon stream successfully read!")

        return {
            "channels": channels,
            "timestamps": timestamps,
            "overflow_types": overflow_types,
            "missed_events": missed_events
        }
    else: #csv option
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        channels=data[:,0]
        timestamps=data[:,1]
        overflow_types=[]
        missed_events=[]

        return {
            "channels": channels,
            "timestamps": timestamps,
            "overflow_types": overflow_types,
            "missed_events": missed_events
        }
def process_photon_data(photon_data):
    channels = photon_data["channels"]
    timestamps = photon_data["timestamps"]
    overflow_types = photon_data["overflow_types"]
    missed_events = photon_data["missed_events"]
    print('ACQUIRED DATA')

    return timestamps, channels
def create_channels(timestamps, channels, channel_names):
    channel_signals = []
    for name in channel_names:
        ch = timestamps[channels == name]
        if len(ch) > 0:
            channel_signals.append(ch)

    if len(channel_signals) == 0:
        raise ValueError("No valid channels found")

    return channel_signals
def make_tau(time_bounds, lag_prec):
    """
    This makes the tau bins. We make exponentially sized cascades, each with lag_prec bins.
    Parameters
    ----------
    time_bounds
    lag_prec

    Returns
    -------
    lags, division_factor, lag_bin_edges
    """

    ''''''
    start_time, stop_time = time_bounds

    '''create log 2 spaced lags'''
    cascade_end = int(np.log2(stop_time))  # cascades are collections of lags  with equal bin spacing 2^cascade
    nper_cascade = lag_prec  # number of equal
    a = np.array([2 ** i for i in range(1, cascade_end + 1)])
    b = np.ones(nper_cascade)
    division_factor = np.kron(a, b)
    lag_bin_edges = np.cumsum(division_factor / 2)

    # find the bin region
    start_bin = np.argmin(np.abs(lag_bin_edges - start_time))
    stop_bin = np.argmin(np.abs(lag_bin_edges - stop_time))
    lag_bin_edges = lag_bin_edges[start_bin:stop_bin + 1]  # bins
    lags = np.sqrt(lag_bin_edges[1:] * lag_bin_edges[:-1])
    division_factor = division_factor[
                      start_bin + 1: stop_bin + 1
                      ]  # normalization factor
    return lags, division_factor, lag_bin_edges
@jit(nopython=True)
def photons_in_bins(ch0, ch1, lag_bin_edges, offset_lag=0):
        """
        Adapted from Boris Spokoyny's code.
        This function allows to correlate the photon-stream on a log timescale. The photon correlation is stored as a property of the photons class: self.cross_corr or self.auto_corr.

        file_in: file ID of the photons-file to correlate
        correlations: 'cross_corr' or 'auto_corr'.
        channels: Hydraharp channels to be correlated. e.g. [0,1] for cross-correlation of channels 0 and 1.
        time_bounds: upper and lower limit for the correlation. In ps for T2, in pulses for T3.
        lag_precision: Number of correlation points between time-spacings of log(2). Must be integers larger than 1.
        lag_offset: offset in ps or pulses between the channels.

        This algorithm computes the cross-correlation between ch0 and ch1 variables.
        For T2 data, ch0 and ch1 correspond to absolute photon arrival times.
        For T3 data, ch0 and ch1 should correspond to the photon arrival sync pulse number.
        start_time and stop_time for T2 data should be in time units of the photon arrival times, and for T3 data should be in units of sync pulses.

        The correlation lags are log(2) spaced with coarseness # of lags per cascade, i.e. if start_time = 1; stop_time = 50; coarseness = 4; the lag bins will be [1, 2, 3, 4;  6, 8, 10, 12;  16, 20, 24, 28;  36, 44 ]. If coarseness = 2, the lag bins become [1, 2;  4, 6;  10, 14;  22,30;  46].
        The algorithm works by taking each photon arrival time and counting the number of photons that are lag bins away. For example say our lag bin edges are [1, 2, 4, 6, 10, 14]
                Time Slot: 1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
                Photon     1   1   0   0   1   1   0   1   1   0   1   0   0   0   1
                1st Photon ^
                Lag bins   |   |       |       |               |               |
                # photons    1     0       2           2                1
                2nd Photon     ^
                Lag bins       |   |       |       |               |               |
                # photons        0     1       1           3               1
                3rd Photon                 ^
                Lag bins                   |   |       |       |               |               |
                # photons                    1     2       1           1               1

                etc..
        The cross-correlation is the sum of all the photons for each bin, i.e. for the three runs above we get [2, 3, 4, 6, 3].

        This function counts the number of photons in the photon stream bins according to a prescription from Ted Laurence: Fast flexible algorithm for calculating photon correlation, Optics Letters, 31, 829, 2006
        """
        num_ch0 = len(ch0)
        num_ch1 = len(ch1)
        n_edges = len(lag_bin_edges)
        low_inds = np.zeros(n_edges, dtype=np.int64) #these indices will help us keep track of the photons in ch1 we have already seen
        max_inds = np.zeros(n_edges, dtype=np.int64)
        acf = np.zeros(n_edges-1)

        low_inds[0] = 0 # changed from starting at 1 which caused negative numbers of photons

        is_auto_corr= (ch0 is ch1) #check first to see if the signals are equivalent

        for phot_ind in range(num_ch0): #we loop through the index of every event in ch0
            # for that particular photon, we get all the bin edges in absolute time. These bins contain the photons we will count
            bin_edges = ch0[phot_ind] + lag_bin_edges + offset_lag
            for k in range(n_edges-1): #for each bin
                # if the channels correspond to absolute time (not sync idx) and are EXACTLY equal
                if ch0[phot_ind] == ch1[low_inds[k]] and is_auto_corr:
                    low_inds[k] += 1  # skip that photon since you don't want to include same events

                while low_inds[k] < num_ch1 and ch1[low_inds[k]] < bin_edges[k]:
                    #scan through ch1 indices which are less than the lower edge of the bin. None of these photons will be included
                    low_inds[k] += 1

                while max_inds[k] < num_ch1 and ch1[max_inds[k]] < bin_edges[k+1]:
                    #scan through ch1 indices which are less than the upper edge of the bin. SOME of these will be included
                    max_inds[k] += 1

                acf[k] += max_inds[k] - low_inds[k] #we include photons that are less than bin_edge[k+1] AND not less than bin_edge[k] (b1 <= p < b2)

                low_inds[k+1] = max_inds[k] #we update the region of search for the next bin.
                # We don't need to scan all the ch1 photons close to our ch0 photon again.
                # We want to look further and we now know where to start
                # although we could update max_inds in after finding low_inds, it turns out this is actually slower than just doing the while loop

            #we can keep the same indices since we move to a large photon in ch0.
            # Since the bins haven't changed, the indices can only grow larger.
            # In other words, we can use the bins of photA to inform on the bins of photB

        return acf
def photon_corr(ch0, ch1, time_bounds, lag_precision, lag_offset=0):

    """

    Parameters
    ----------
    ch0
    ch1
    time_bounds
    lag_precision
    lag_offset

    Returns
    -------
    corr_final, unnormalized_corr
    """
    time_start = timing.time()

    start_time, stop_time = time_bounds

    '''create log 2 spaced lags'''
    cascade_end = int(np.log2(stop_time))  # cascades are collections of lags  with equal bin spacing 2^cascade
    nper_cascade = lag_precision  # number of equal
    a = np.array([2 ** i for i in range(1, cascade_end + 1)])
    b = np.ones(nper_cascade)
    division_factor = np.kron(a, b)
    lag_bin_edges = np.cumsum(division_factor / 2)

    # find the bin region
    start_bin = np.argmin(np.abs(lag_bin_edges - start_time))
    stop_bin = np.argmin(np.abs(lag_bin_edges - stop_time))
    lag_bin_edges = lag_bin_edges[start_bin:stop_bin + 1]  # bins
    lags = np.sqrt(lag_bin_edges[1:] * lag_bin_edges[:-1])
    division_factor = division_factor[start_bin + 1:stop_bin + 1]  # normalization factor

    # counters etc for normalization
    ch0_min = np.inf
    ch1_min = np.inf  # minimum time tag
    ch0_count = len(ch0)
    ch1_count = len(ch1)  # photon numbers in each channel
    print('Number of photons in ch0 is', len(ch0))
    print('Number of photons in ch1 is', len(ch1))
    ch0_min = min(ch0_min, min(ch0))
    ch1_min = min(ch1_min, min(ch1))

    '''correlating '''
    tic = timing.time()
    print('Correlating data...\n')

    corr = photons_in_bins(ch0, ch1, lag_bin_edges, lag_offset)

    # normalization
    ch0_max = max(ch0)
    ch1_max = max(ch1)
    print('Ch0 max is', ch0_max)
    print('Ch1 max is', ch1_max)
    tag_range = max(ch1_max, ch0_max) - min(ch1_min, ch0_min)  # range of tags in the entire dataset
    infinite_acq_correction = tag_range/(tag_range - lags)  # has greater effect at long tau
    unnormalized_corr = corr * infinite_acq_correction  # if your experiment was infinitely long with proportional count rate, how many correlation events in each bin

    # intensity_normalized_corr = unnormalized_corr * ch0_max / (ch0_count * ch1_count) #this is practically identical to the numerator in Laurence
    intensity_normalized_corr = unnormalized_corr * tag_range / (ch0_count * ch1_count)

    corr_final = 2*intensity_normalized_corr / division_factor #assume 1,0 comparisons are identical

    print('Done\n')
    toc = timing.time()
    print('Time elapsed during correlating is %4f s' % (toc - tic))

    time_end = timing.time()
    print('Total time elapsed is %4f s' % (time_end - time_start))

    return corr_final, unnormalized_corr
def add_signals(listed_channels):
    return np.array(np.sort(np.concatenate(listed_channels)))
def undo_create_channels(channel_signals, channel_names):
    # Concatenate timestamps
    timestamps = np.concatenate(channel_signals)

    # Create matching channel labels
    channels = np.concatenate([
        np.full(len(ch), name)
        for ch, name in zip(channel_signals, channel_names)
    ])

    # Restore global chronological order
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    channels = channels[order]

    return timestamps, channels
def truncate_photon_stream(timestamps, channels, start_time, end_time):

    """

    Parameters
    ----------
    timestamps
    channels
    start_time (in seconds)
    end_time (in seconds)

    Returns
    -------
    timestamps
    channels

    """

    local_timestamps=timestamps.copy()
    shift=local_timestamps[0]
    local_timestamps-=shift
    start_time = int(start_time)
    end_time = int(end_time)

    if end_time != -1:
        end_time = int(end_time*1e12)
        mask = local_timestamps <= end_time

        local_timestamps = local_timestamps[mask]
        channels = channels[mask]
    if start_time != 0:
        start_time = int(start_time*1e12)
        mask = local_timestamps >= start_time

        local_timestamps = local_timestamps[mask]
        channels = channels[mask]

    local_timestamps+=shift

    return local_timestamps, channels
def truncate_channel_signals(channel_signals, end_time):
    end_time_in_ps = end_time * 1e12

    new_channel_signals=[]

    if end_time > 0:
        for ch in channel_signals:
            ch=ch[ch<=end_time_in_ps]
            new_channel_signals.append(ch)
        print('Truncated!')
        return new_channel_signals
    else:
        print('No truncation performed')
        return  channel_signals
#Intensity trace code
def find_aggs(cps, mode, input_threshold):
    # 2. Identify spikes
    if mode == 'MEAN-STD':
        threshold = np.mean(cps) + input_threshold * np.std(cps)
    elif mode == 'MED-STD':
        threshold = np.median(cps) + input_threshold * np.std(cps)
    elif mode == 'ABS':
        threshold = input_threshold
    else:
        print('No threshold provided')
        raise TypeError

    agg_bins = np.where(cps > threshold)[0]

    agg_intervals = []

    if len(agg_bins) == 0:
        print("No aggregates detected.")

    # 3. Group contiguous bins into intervals
    if len(agg_bins) > 0:
        start_idx = agg_bins[0]
        for i in range(1, len(agg_bins)):
            if agg_bins[i] != agg_bins[i - 1] + 1:
                # Store indices rather than times to ensure we get edges right
                agg_intervals.append((start_idx, agg_bins[i - 1]))
                start_idx = agg_bins[i]
        agg_intervals.append((start_idx, agg_bins[-1]))
    print(f"Detected {len(agg_intervals)} aggregate events")

    plt.plot(cps)
    plt.axhline(threshold, color='k', ls='--')
    for interval in agg_intervals:

        plt.axvline(interval[0], color='black', alpha=0.5)
        plt.axvline(interval[1], color='red', alpha=0.5)
        print(interval[0])
        print(interval[1])
    plt.show()


    return agg_intervals
def merge_aggs(agg_intervals, d=5):
    merged_intervals = []

    if len(agg_intervals) > 0:
        # Initialize with the first interval
        curr_start, curr_end = agg_intervals[0]

        for i in range(1, len(agg_intervals)):
            next_start, next_end = agg_intervals[i]

            # Check if the gap (next_start - curr_end) is <= d
            # Note: if d=1, they are combined if they are literally touching
            if next_start - curr_end <= d:
                # Extend the current interval to the end of the next one
                curr_end = next_end
            else:
                # Gap is too large; save the current interval and move to the next
                merged_intervals.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end

        # Don't forget to add the last one processed
        merged_intervals.append((curr_start, curr_end))

    print(f"Combined into {len(merged_intervals)} aggregate events (spacing threshold d={d})")

    return merged_intervals
def remove_aggs(timestamps, channels, binwidth, agg_intervals):

    if np.all(agg_intervals == 0):
        return timestamps, channels
    
    timestamps_copy = timestamps.copy()
    channels_copy = channels.copy()
    
    ch_min = np.min(timestamps_copy)

    print('Removing Aggregates...')
    # Remove and shift (Iterating BACKWARDS to keep timestamps_copy valid)
    for start_bin, end_bin in reversed(agg_intervals):
        # Convert bin indices to exact edges in timestamp units
        agg_start = ch_min + (start_bin * binwidth)
        agg_end = ch_min + ((end_bin + 1) * binwidth)
        agg_duration = agg_end - agg_start

        # Create mask for data to keep
        keep_mask = (timestamps_copy < agg_start) | (timestamps_copy > agg_end)
        after_mask = timestamps_copy > agg_end

        # Shift everything occurring after the spike
        timestamps[after_mask] -= int(agg_duration)

        # Remove the spike data
        timestamps_copy = timestamps_copy[keep_mask]
        channels_copy = channels_copy[keep_mask]

    return timestamps_copy, channels_copy
def remove_aggs_CCD(timestamps, data_cube, binwidth, agg_intervals):
    if np.all(agg_intervals == 0):
        return timestamps, data_cube

    timestamps_copy = timestamps.copy()
    data_cube_copy = data_cube.copy()

    ch_min = np.min(timestamps_copy)

    print('Removing Aggregates...')
    # Remove and shift (Iterating BACKWARDS to keep timestamps_copy valid)
    for start_bin, end_bin in reversed(agg_intervals):
        # Convert bin indices to exact edges in timestamp units
        agg_start = ch_min + (start_bin * binwidth)
        agg_end = ch_min + ((end_bin + 1) * binwidth)
        agg_duration = agg_end - agg_start

        # Create mask for data to keep
        keep_mask = (timestamps_copy < agg_start) | (timestamps_copy > agg_end)
        after_mask = timestamps_copy > agg_end

        # Shift everything occurring after the spike
        timestamps_copy[after_mask] -= agg_duration

        # Remove the spike data
        timestamps_copy = timestamps_copy[keep_mask]
        data_cube_copy = data_cube_copy[keep_mask]

    return timestamps, data_cube
def find_and_remove_aggs(timestamps, channels, binwidth, mode, input_threshold, plot=True):

    if input_threshold==0:
        return timestamps, channels, [[0,0]]

    bin_centers, cps, _ = calculate_intensity_trace(timestamps, binwidth)

    agg_intervals = find_aggs(cps, mode, input_threshold)
    agg_intervals = merge_aggs(agg_intervals, d=5)

    print(agg_intervals)

    clean_timestamps, clean_channels = remove_aggs(timestamps, channels, binwidth, agg_intervals)

    if plot:
        plt.plot(bin_centers, cps, label='Original')
        new_bin_centers, new_cps, _ = calculate_intensity_trace(clean_timestamps, binwidth)
        plt.plot(new_bin_centers, new_cps, label='New')
        plt.xlabel('Time (s)')
        plt.ylabel('Counts/sec')
        plt.legend()
        plt.show()

    return clean_timestamps, clean_channels, agg_intervals
@jit(nopython=True)
def calculate_intensity_trace(ch, binwidth):
    """
    Numba-compatible intensity trace without edge artifacts.

    Parameters
    ----------
    ch: list/array of times in ps
    binwidth: in ps

    Returns
    -------
    time, counts_per_second, avg_cps
    """
    ch_min = np.min(ch)
    ch_max = np.max(ch)

    # Number of FULL bins only
    num_bins = int((ch_max - ch_min) // binwidth)

    counts = np.zeros(num_bins, dtype=np.int64)

    for i in range(ch.shape[0]):
        bin_index = int((ch[i] - ch_min) // binwidth)
        if bin_index >= num_bins:
            continue  # skip last partial bin
        counts[bin_index] += 1

    # Bin centers in seconds
    time = np.empty(num_bins, dtype=np.float64)
    for j in range(num_bins):
        time[j] = (ch_min + (j + 0.5) * binwidth) * 1e-12


    cps = counts / (binwidth / 1e12)
    avg_cps = np.mean(cps)

    return time, cps, avg_cps
def assess_intensity_stability(time, cps, threshold=0.9):
    threshold_value = cps[0] * threshold
    threshold_time = time[get_index(cps, threshold_value)]

    # Setup the plot
    fig, ax = plt.subplots()
    ax.plot(time, cps)
    ax.axhline(threshold_value, color='black', ls='--')
    ax.axvline(threshold_time, color='black', ls='--')
    ax.set_ylim(bottom=0, top=max(cps) * 1.5)

    # CRITICAL: Add explicit padding to the X-axis so you can easily click
    # "before" time[0] and "after" time[-1]
    padding = (time[-1] - time[0]) * 0.05  # 5% padding
    ax.set_xlim(left=time[0] - padding, right=time[-1] + padding)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity (counts/bin)')

    # Instruct the user via the title
    ax.set_title("Click twice: 1st for START time, 2nd for END time.\nClick in empty margins for full range (0 or -1).")
    plt.draw()

    # Capture exactly 2 clicks from the user
    clicks = plt.ginput(n=2, timeout=0)
    plt.close(fig)  # Close the window after getting the clicks

    # Extract the X-coordinates (Time)
    clicked_start = clicks[0][0]
    clicked_end = clicks[1][0]

    # Evaluate start time
    if clicked_start <= time[0]:
        start_time = 0
        print(f'Start time set to: 0 (Data starts at {time[0]:.1f} s)')
    else:
        start_time = int(clicked_start)
        print(f'Start time set to: {start_time} s')

    # Evaluate end time
    if clicked_end >= time[-1]:
        end_time = -1
        print(f'End time set to: -1 (Data ends at {time[-1]:.1f} s)')
    else:
        end_time = int(clicked_end)
        print(f'End time set to: {end_time} s')  # Fixed typo here

    return start_time, end_time
def get_slices(timestamps):
    binwidth = timestamps[-1] / 1000
    time, cps, avg_cps = calculate_intensity_trace(timestamps, binwidth=binwidth)

    fig, ax = plt.subplots()
    ax.plot(time, cps)
    ax.set_ylim(bottom=min(cps) * 0.75, top=max(cps) * 1.25)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity (counts/bin)')

    ax.set_title("Click transition edges. Press ENTER / Middle-Click when done.")
    plt.draw()

    # n=-1 allows an arbitrary number of clicks until the user presses Enter
    clicks = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    # Extract just the X-coordinates (times) of the transitions
    sub_interval_edges = [click[0] for click in clicks]

    if not sub_interval_edges:
        sub_interval_edges=[-1]

    print(sub_interval_edges)

    return sub_interval_edges
def plot_intensity_trace(time, counts_per_bin, avg_cps):
    plt.plot(time, counts_per_bin)
    plt.ylim(bottom=min(counts_per_bin)*.5, top=max(counts_per_bin)*1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (counts/bin)')
    plt.title(f'Intensity Trace (avg={avg_cps:.0f} cps)')
    plt.show()
def threshold_intensity(threshold_channel, all_channels, binwidth,
                        lower=0, upper=np.inf):

    #Compute intensity trace
    t, counts, _ = calculate_intensity_trace(
        threshold_channel, binwidth=binwidth
    )

    #determine which bins are valid
    valid_bins = (counts >= lower) & (counts <= upper)

    t_start = t[0] - binwidth / 2
    num_bins = len(valid_bins)

    def filter_channel(channel):
        channel = np.asarray(channel)
        if channel.size == 0:
            return channel

        indices = ((channel - t_start) // binwidth).astype(int)

        in_range = (indices >= 0) & (indices < num_bins)

        keep = np.zeros(len(channel), dtype=bool)
        keep[in_range] = valid_bins[indices[in_range]]

        return channel[keep]

    return [filter_channel(ch) for ch in all_channels]
#Lifetime code
def lifetime(start_events, stop_events,t_rep, n_bins=1000, gate=0, plot=False, verbose=False):

    """

    Parameters
    ----------
    t_rep (in ps)
    start_events
    stop_events
    plot
    gate
    n_bins

    Returns
    -------
    bin_centers
    counts

    """
    if verbose:
        print('Num Syncs:', len(start_events))
        print('Num Photons:', len(stop_events))

    if verbose:
        print('Num Syncs (filtered):', len(start_events))
        print('Num Photons (filtered):', len(stop_events))
    # calculate microtimes
    time_seps = get_lifetime(start_events, stop_events)
    #use remainder from t_rep so we don't lose photons from the lifetime alignment
    time_seps = time_seps % t_rep

    bins = np.linspace(0, t_rep, n_bins + 1)

    counts, edges = np.histogram(time_seps, bins=bins)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    counts = counts[bin_centers >= gate]
    bin_centers = bin_centers[bin_centers >= gate]

    # Normalize bin centers to the peak
    # max_index = np.argmax(counts)
    # bin_centers -= bin_centers[max_index]
    # bin_centers+=gate
    bin_centers /= 1e3  # Convert to nanoseconds

    if plot:
        plt.figure()
        plt.plot(bin_centers, counts)
        plt.xlabel("Time (ns)")
        plt.ylabel("Counts")
        plt.title("Lifetime")
        plt.yscale('log')
        plt.show()

    return bin_centers, counts
def get_lifetime(start_events, stop_events):
    """

    Parameters
    ----------
    start_events
    stop_events

    Returns
    -------
    lifetime

    """

    start_events = np.asarray(start_events)
    stop_events = np.asarray(stop_events)

    # for each stop, find latest start before it
    start_idx = np.searchsorted(start_events, stop_events, side='right') - 1

    valid = start_idx >= 0

    lifetime = stop_events[valid] - start_events[start_idx[valid]]

    return lifetime
def remove_afterpulses(sync_times, det_times):
    '''
    Written by JH.

    Parameters
    ----------
    sync_times
    det_times

    Returns
    -------
    filtered_det_times
    filtered_syncs

    '''
    # --------------------------------------------------
    # Assign each photon to its preceding sync pulse
    # --------------------------------------------------
    sync_idx = np.searchsorted(sync_times, det_times, side="right") - 1

    # Discard photons that occur before the first sync
    valid = sync_idx >= 0
    det_times = det_times[valid]
    sync_idx = sync_idx[valid]

    # --------------------------------------------------
    # Remove afterpulses (keep first photon per sync)
    # --------------------------------------------------
    _, first_indices = np.unique(sync_idx, return_index=True)

    filtered_det_times = det_times[first_indices]
    filtered_syncs = sync_times[sync_idx[first_indices]]

    return filtered_syncs, filtered_det_times
def get_lifetime_offsets(sync_times, det_channel_signals, t_rep, n_bins=100000, plot=True):

    peak_times = []
    for ch in det_channel_signals:
        x, y = lifetime(sync_times, ch, t_rep, n_bins=n_bins)

        start_idx = get_index(x, 40)
        end_idx = get_index(x, 100)

        baseline = np.mean(y[start_idx:end_idx])

        max_index = np.argmax(y)
        peak_time = x[max_index]

        half_max = (max(y) - baseline) / 50 + baseline

        # Find the points where the fit crosses half of the maximum intensity
        left_idx = np.where(y >= half_max)[0][0]  # First crossing
        rise_time = x[left_idx]

        if plot:
            plt.semilogy(x, normalize_lifetime_for_log_plot(x, y, baseline))
            plt.axvline(peak_time, color='k', alpha=0.5, ls='--')
            plt.axvline(rise_time, color='k', alpha=0.5, ls=':')

        peak_times.append(rise_time * 1e3)  # convert to ps

    if plot:
        plt.xlabel('$\\tau$ (ns)')
        plt.ylabel('Intensity (norm.)')
        plt.title('Original')
        plt.show()

    shifts = np.rint(np.array(peak_times) - peak_times[0]).astype(int)
    print('------Shifts in ps------')
    for i, shift in enumerate(shifts):
        print(f'shift{i} = {shift}')

    if plot:
        for i, ch in enumerate(det_channel_signals):
            ch = ch - shifts[i]
            x, y = lifetime(sync_times, ch, t_rep, n_bins=n_bins)

            start_idx = get_index(x, 40)
            end_idx = get_index(x, 100)

            baseline = np.mean(y[start_idx:end_idx])
            plt.semilogy(x, normalize_lifetime_for_log_plot(x, y, baseline))

            max_index = np.argmax(y)
            peak_time = x[max_index]

            half_max = (max(y) - baseline) / 50 + baseline

            # Find the points where the fit crosses half of the maximum intensity
            left_idx = np.where(y >= half_max)[0][0]  # First crossing
            rise_time = x[left_idx]

            plt.axvline(peak_time, color='k', alpha=0.5, ls='--')
            plt.axvline(rise_time, color='k', alpha=0.5, ls=':')
        plt.xlabel('$\\tau$ (ns)')
        plt.ylabel('Intensity (norm.)')
        plt.title('Shifted')
        plt.show()

    return shifts
def normalize_lifetime_for_log_plot(x, counts, baseline, min_val=1e-3):

    zeroed_lt = counts - baseline

    # 4. Min-Max Scale to a strict range (e.g., 10^-3 to 1) so semilogy looks perfect
    # This prevents log(0) errors while forcing baselines to match.
    max_val = 1.0

    # Formula: ((x - min) / (max - min)) scaled to our desired log floor
    peak = np.max(zeroed_lt)

    normalized_lt = min_val + (zeroed_lt / peak)

    return normalized_lt
def trim_data_for_lifetime(x, y):
    """
    Sets peak of decay to zero time, removes flat baseline before decay

    Parameters
    ----------
    x: time bins
    y: counts

    Returns
    -------
    x, y: trimmed bins, counts

    """
    peak_index = np.argmax(y)

    new_x = x.copy()[peak_index:]
    new_y = y.copy()[peak_index:]

    shift=min(new_x)

    new_x -= shift

    return new_x, new_y, shift
def compare_rep_rates(bin_centers, counts, num_gaussian):

    fitting_x, fitting_y, shift = trim_data_for_lifetime(bin_centers, counts)


    current_popt, fit = fit_lifetime_n_exp(fitting_x, fitting_y, n=num_gaussian, plot=True)

    half_x = np.linspace(fitting_x[0], fitting_x[-1] / 2, int(len(fitting_x) / 2))
    double_x = np.linspace(fitting_x[0], fitting_x[-1] * 2, 2 * len(fitting_x))

    shorter_fit = n_exponential(half_x, *current_popt)
    longer_fit = n_exponential(double_x, *current_popt)

    plt.semilogy(fitting_x, fitting_y)
    plt.semilogy(half_x, shorter_fit)
    plt.semilogy(fitting_x, fit)
    plt.semilogy(double_x, longer_fit)
    plt.show()

    baseline = current_popt[-1]

    fitting_y -= baseline
    longer_fit -= baseline
    fit -= baseline
    shorter_fit -= baseline

    plt.semilogy(fitting_x, fitting_y)
    plt.semilogy(fitting_x, fit)
    plt.semilogy(double_x, longer_fit)
    plt.semilogy(half_x, shorter_fit)
    plt.show()

    long_intensity = np.trapz(longer_fit, double_x)
    short_intensity = np.trapz(shorter_fit, half_x)
    true_intensity = np.trapz(fit, fitting_x)

    print(f'Current Rep Period vs Double Rep Period {true_intensity / long_intensity:.4f}')
    print(f'Half Rep Period vs Double Rep Period {short_intensity / long_intensity:.4f}')
def time_gate(start_events, stop_events,t_rep, gate=0.0):
    filtered_start_events = []
    filtered_stop_events = []

    # Pointer for stop events
    stop_index = 0

    # Iterate over each start event
    for start_time in start_events:
        # Find the first stop event after the start event
        # If a valid stop event exists, add the pair to the filtered lists
        while stop_index < len(stop_events) and stop_events[stop_index] < start_time + gate:
            stop_index += 1

        # If a valid stop event exists within the optional repetition period
        if stop_index < len(stop_events) and stop_events[stop_index] <= start_time + t_rep:
            filtered_start_events.append(start_time)
            filtered_stop_events.append(stop_events[stop_index])

    return np.array(filtered_start_events), np.array(filtered_stop_events)
def pulsed_g2(ch0, ch1, center=0, time_range=600e3, n_bins=100, normalize=False, plot=True, pos_only=False):
    # Calculate the width of a single bin
    bin_width = time_range // n_bins

    # 1. Generate clean, counterfeited bin edges based on directional choice
    pos_edges = np.arange(0, time_range + bin_width, bin_width)

    if pos_only:
        lag_bin_edges = pos_edges + center
    else:
        # Create negative side edges (excluding 0 to avoid duplication) and flip them
        neg_edges = -pos_edges[1:][::-1]
        # Combine them and shift by the center offset
        lag_bin_edges = np.concatenate((neg_edges, pos_edges)) + center

    # 2. Calculate bin centers (lags) for plotting/indexing based on the edges
    g2_lags = (lag_bin_edges[:-1] + bin_width / 2) / 1e3  # converted to ns

    # Run the correlation function
    print('Calculating pulsed g2...')
    corr = photons_in_bins(ch0, ch1, lag_bin_edges=lag_bin_edges, offset_lag=0)

    if normalize:
        corr = corr / max(corr)

    g2 = corr
    total_bins = len(g2)

    # 3. Dynamic Peak Integration Slicing
    if not pos_only:
        # Symmetric window: [Side Peak 1 | Center Peak | Side Peak 2]
        cutoff1 = int(total_bins / 3)
        cutoff2 = 2 * cutoff1

        print(f"Cutoff indices: {cutoff1}, {cutoff2}")
        print(f"Cutoff times (ns): {g2_lags[cutoff1]:.2f}, {g2_lags[cutoff2]:.2f}")

        side_peak1_heights = g2[:cutoff1]
        center_peaks_heights = g2[cutoff1:cutoff2]
        side_peak2_heights = g2[cutoff2:]

        side_peak1_area = np.sum(side_peak1_heights)
        center_peak_area = np.sum(center_peaks_heights)
        side_peak2_area = np.sum(side_peak2_heights)

        print(f"Areas -> Side1: {side_peak1_area}, Center: {center_peak_area}, Side2: {side_peak2_area}")

        # Standard ratio for symmetric data: 2 * Center / (Left + Right)
        ratio = 2 * center_peak_area / (side_peak1_area + side_peak2_area)
        print('BX/X Ratio (Symmetric):', ratio)

    else:
        # Positive-only window: [Center Peak (Right Half) | Side Peak 1]
        # Split the positive array in half to separate the zero-delay peak from the first side peak
        cutoff = int(total_bins / 2)

        print(f"Cutoff index: {cutoff}")
        print(f"Cutoff time (ns): {g2_lags[cutoff]:.2f}")

        center_peaks_heights = g2[:cutoff]  # This is the right-half of the zero-delay peak
        side_peak_heights = g2[cutoff:]  # This is the first positive side peak

        center_peak_half_area = np.sum(center_peaks_heights)
        side_peak_area = np.sum(side_peak_heights)

        print(f"Areas -> Center (half-peak): {center_peak_half_area}, Side Peak: {side_peak_area}")

        # Because center_peaks_heights only captures the right half of the zero-delay peak,
        # we multiply it by 2 to reconstruct the "full" center peak area for an accurate ratio.
        # Ratio = Full Center Area / Single Side Area
        ratio = (2 * center_peak_half_area) / side_peak_area
        print('BX/X Ratio (Pos-Only):', ratio)

    if plot:
        plt.plot(g2_lags, g2, '-o', markersize=1)
        plt.ylabel('Event counts')
        plt.xlabel('Pulse separation [ns]')
        # if not pos_only:
        #     plt.axvline(g2_lags[cutoff1], color='r', linestyle='--', alpha=0.5)
        #     plt.axvline(g2_lags[cutoff2], color='r', linestyle='--', alpha=0.5)
        # else:
        #     plt.axvline(g2_lags[cutoff], color='r', linestyle='--', alpha=0.5)
        plt.show()

    return g2_lags, g2, ratio
def sliced_photon_corr(ch0, ch1, time_bounds, lag_prec, slice_len_in_s):

    slice_len_in_ps=slice_len_in_s*1e12

    local_exp_length=max(max(ch0),max(ch1))

    num_slices=int(np.ceil(local_exp_length/slice_len_in_ps))

    final_corr=np.zeros(len(make_tau(time_bounds, lag_prec)[0]))

    for i in range(num_slices):

        start_time=i*slice_len_in_ps
        end_time=(i+1)*slice_len_in_ps

        ch0_mask= (ch0>= start_time) & (ch0 < end_time)
        ch1_mask= (ch1>= start_time) & (ch1 < end_time)

        local_ch0=ch0[ch0_mask]-start_time
        local_ch1=ch1[ch1_mask]-start_time

        _, local_unnorm_corr=photon_corr(local_ch0, local_ch1, time_bounds, lag_prec)

        final_corr+=local_unnorm_corr


    tag_range=max(max(ch0), max(ch1))-min(min(ch0),min(ch1))
    intensity_correction= tag_range / (len(ch0) * len(ch1))
    # final_corr/=num_slices

    final_corr*=2*intensity_correction

    return final_corr
def make_tau_neg(time_bounds, lag_precision):

    time_bounds=np.array(time_bounds)
    if (time_bounds<0).any():
        switch_sign=True
        time_bounds = -time_bounds
    else:
        switch_sign=False

    start_time,stop_time=np.sort(time_bounds)

    if (start_time > 0 and stop_time < 0) or (start_time < 0 and stop_time > 0):
        raise ValueError("Start time and stop time must have the same sign (both positive or both negative).")


    """create log 2 spaced lags"""
    cascade_end = int(
        np.log2(stop_time)
    )  # cascades are collections of lags  with equal bin spacing 2^cascade
    nper_cascade = lag_precision  # number of equal
    a = np.array([2 ** i for i in range(1, cascade_end + 1)])
    b = np.ones(nper_cascade)

    division_factor = np.kron(a, b)
    lag_bin_edges = np.cumsum(division_factor / 2)
    lags = (lag_bin_edges[:-1] + lag_bin_edges[1:]) * 0.5

    # find the bin region
    start_bin = np.argmin(np.abs(lag_bin_edges - start_time))
    stop_bin = np.argmin(np.abs(lag_bin_edges - stop_time))
    lag_bin_edges = lag_bin_edges[start_bin: stop_bin + 1]  # bins
    lags = lags[start_bin + 1: stop_bin + 1]  # center of the bins
    division_factor = division_factor[
                      start_bin + 1: stop_bin + 1
                      ]  # normalization factor

    if switch_sign:
        lags=-lags
        division_factor=division_factor[::-1]

    return lags, division_factor
def photon_corr_neg(ch0, ch1, time_bounds, lag_precision, lag_offset=0):
    time_start = timing.time()

    time_bounds = np.array(time_bounds)
    if (time_bounds < 0).any():
        switch_sign = True
        time_bounds = -time_bounds
    else:
        switch_sign = False

    start_time, stop_time = np.sort(time_bounds)

    if (start_time > 0 and stop_time < 0) or (start_time < 0 and stop_time > 0):
        raise ValueError("Start time and stop time must have the same sign (both positive or both negative).")

    """create log 2 spaced lags"""
    cascade_end = int(
        np.log2(stop_time)
    )  # cascades are collections of lags  with equal bin spacing 2^cascade
    nper_cascade = lag_precision  # number of equal
    a = np.array([2 ** i for i in range(1, cascade_end + 1)])
    b = np.ones(nper_cascade)

    division_factor = np.kron(a, b)
    lag_bin_edges = np.cumsum(division_factor / 2)
    lags = (lag_bin_edges[:-1] + lag_bin_edges[1:]) * 0.5

    # find the bin region
    start_bin = np.argmin(np.abs(lag_bin_edges - start_time))
    stop_bin = np.argmin(np.abs(lag_bin_edges - stop_time))
    lag_bin_edges = lag_bin_edges[start_bin: stop_bin + 1]  # bins
    lags = lags[start_bin + 1: stop_bin + 1]  # center of the bins
    division_factor = division_factor[
                      start_bin + 1: stop_bin + 1
                      ]  # normalization factor

    if switch_sign:
        lags = -lags
        # division_factor = division_factor[::-1]
        division_factor = division_factor
        lag_bin_edges =lag_bin_edges

    # plt.plot(lags, lag_bin_edges[:-1])
    # plt.show()
    # counters etc for normalization
    ch0_min = np.inf
    ch1_min = np.inf  # minimum time tag
    ch0_count = len(ch0)
    ch1_count = len(ch1)  # photon numbers in each channel
    print('Number of photons in ch0 is', len(ch0))
    print('Number of photons in ch1 is', len(ch1))
    ch0_min = min(ch0_min, min(ch0))
    ch1_min = min(ch1_min, min(ch1))

    '''correlating '''
    tic = timing.time()
    print('Correlating data...\n')

    corr = photons_in_bins(ch0, ch1, lag_bin_edges, lag_offset)

    print('Length of correlation', len(corr))

    # normalization
    ch0_max = max(ch0)
    ch1_max = max(ch1)
    print('Ch0 max is', ch0_max)
    print('Ch1 max is', ch1_max)
    tag_range = max(ch1_max, ch0_max) - min(ch1_min, ch0_min)  # range of tags in the entire dataset


    print(max(lag_bin_edges), min(lag_bin_edges))
    print(max(lags), min(lags))
    corr_div = corr / division_factor
    # plt.plot(corr)
    # plt.plot(division_factor)
    # plt.plot(corr_div)
    # plt.show()

    corr_norm = 2 * corr_div * tag_range / (tag_range - lags) * ch0_max / (ch0_count * ch1_count)
    print('Done\n')
    toc = timing.time()
    print('Time elapsed during correlating is %4f s' % (toc - tic))

    time_end = timing.time()
    print('Total time elapsed is %4f s' % (time_end - time_start))

    return corr_norm
def focus_exp_region(timestamps, channels):
    #plots the experimental time trace and allows user to input the interval of interest
    binwidth = timestamps[-1] / 1000  # calculate coarse intensity trace to isolate important region
    time, cps, avg_cps = calculate_intensity_trace(timestamps, binwidth=binwidth)

    plt.plot(time, cps)
    plt.ylim(bottom=min(cps) * .5, top=max(cps) * 1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (counts/bin)')
    plt.title(f'Intensity Trace (avg={avg_cps:.0f} cps)')
    plt.show()

    start_time = input(
        'Choose the experimental start time (in integer sec.) based on the intensity trace. \n'
        'Enter 0 if no truncation needed: ').lower()
    end_time = input(
        'Choose the experimental end time (in integer sec.) based on the intensity trace. \n'
        'Enter -1 if no truncation needed: ').lower()

    start_time = int(start_time)
    end_time = int(end_time)

    if end_time!=-1:
        end_time*=1e12
        mask=timestamps<=end_time

        timestamps=timestamps[mask]
        channels = channels[mask]
    if start_time!=0:
        start_time*=1e12
        mask=timestamps>=start_time

        timestamps=timestamps[mask]
        channels = channels[mask]

    return timestamps, channels
def extract_precise_period(sync_timestamps, approx_laser_period, window=1000):
    """
    Find distribution of peak shifts between two laser pulses. Searches around the approx. rep period

    Parameters
    ----------
    sync_timestamps
    approx_period_ps: from laser specs
    window_ps: how far on either side of the peak to look (e.g., +/- 500 ps)

    Returns
    -------
    precise_period (in ps)
    """

    # 1. Calculate raw differences again
    diffs = np.diff(sync_timestamps)

    # 2. Filter out ONLY the differences that fall inside our first peak
    min_val = approx_laser_period - window
    max_val = approx_laser_period + window

    first_peak_diffs = diffs[(diffs > min_val) & (diffs < max_val)]

    if len(first_peak_diffs) == 0:
        print("No data found in that window! Check your approx_period_ps.")
        return

    # 3. Calculate the exact mean (this is your highly precise period)
    precise_period = np.mean(first_peak_diffs)
    std_dev = np.std(first_peak_diffs)

    # 4. Plot just this isolated peak to verify it looks clean
    plt.figure(figsize=(8, 4))
    plt.hist(first_peak_diffs, bins=50, color='mediumseagreen', edgecolor='black')
    plt.axvline(precise_period, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {precise_period:.2f} ps')

    plt.title("Isolated First Peak")
    plt.xlabel("Time Difference (ps)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()

    print(f"Precise Laser Period: {precise_period:.3f} ps")
    print(f"Peak Width (Std Dev): {std_dev:.3f} ps")
    print(f"Number of consecutive photon pairs used: {len(first_peak_diffs)}")

    return int(precise_period)

def convolve_lifetime_w_irf(signal_x, signal_y, irf_x, irf_y, plot=True):

    from scipy.signal import convolve
    dt = signal_x[1] - signal_x[0]

    # --- FIX: Prevent edge drop-offs via constant edge padding ---
    # We pad by the length of the IRF to ensure the convolution has enough
    # real data to slide over at the boundaries.
    pad_len = len(irf_y)

    pad_left = np.full(pad_len, min(signal_y))
    pad_right = np.full(pad_len, signal_y[-1])
    padded_signal_y = np.concatenate([pad_left, signal_y, pad_right])

    # Convolve the padded signal
    full_conv_padded = convolve(padded_signal_y, irf_y, mode='full') * dt

    # Strip the padding out.
    # Because of convolution linearity, stripping exactly pad_len from both ends
    # yields an array of the exact same length as an unpadded mode='full' convolution.
    # This means your alignment logic below works perfectly without any changes!
    full_conv = full_conv_padded[pad_len: -pad_len]

    if plot:
        plt.semilogy(full_conv)
        plt.title('Convolution (check a portion looks correct)')
        plt.show()

    #we now get the portion which contains the data we care about
    data_peak = np.argmax(signal_y)
    conv_peak = np.argmax(full_conv)

    # Determine the slice of full_conv that aligns its peak with the measured data peak
    start_idx = conv_peak - data_peak
    end_idx = start_idx + len(signal_x)

    # Create an empty array of NaNs to avoid y=0 log scale errors
    convolution = np.full(len(signal_x), np.nan)

    # Find the valid bounds to prevent indexing errors
    slice_start = max(0, start_idx)
    slice_end = min(len(full_conv), end_idx)
    dest_start = max(0, -start_idx)
    dest_end = dest_start + (slice_end - slice_start)

    # Insert the valid convolved data into our NaN array
    convolution[dest_start:dest_end] = full_conv[slice_start:slice_end]

    irf_norm=normalize_lifetime_for_log_plot(irf_x, irf_y, baseline=np.mean(irf_y[-20:-10]))
    signal_norm=normalize_lifetime_for_log_plot(signal_x, signal_y, baseline=np.mean(signal_y[-20:-10]))
    conv_norm=normalize_lifetime_for_log_plot(signal_x, convolution, baseline=np.mean(convolution[-20:-10]))


    # --- Plotting ---
    if plot:
        plt.figure(figsize=(8, 6))
        plt.semilogy(irf_x, irf_norm, color='k', label='IRF')
        plt.semilogy(signal_x, signal_norm, label='Measured Data', alpha=0.8)
        plt.semilogy(signal_x, conv_norm, '--', color='purple', label='Convolution')
        plt.ylim(bottom=1e-4) # Adjust this based on where your noise floor is
        plt.legend()
        plt.show()

    return convolution
import types
__all__ = [name for name, obj in globals().items()
           if not name.startswith('_') and isinstance(obj, types.FunctionType)]