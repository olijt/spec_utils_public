import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import tkinter as tk
from tkinter import filedialog

'''Saving functions'''
def OT_csv_spectrum_reader(path):
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
    except ValueError:
        data = np.loadtxt(path, delimiter=',', skiprows=2)

    wavelength=data[:,0]
    spectrum=data[:,1]
    if len(wavelength)<5:
        wavelength=data[0]
        spectrum=data[1]

    return wavelength, spectrum

def save_stacked_data_to_csv(save_path, stacked_data, header=""):
    np.savetxt(save_path, stacked_data, delimiter=",", header=header)
    print('\n-------------\n SAVED\n -------------\n ')

def OT_flexible_file_reader(file_path=None):
    import tkinter as tk
    from tkinter import filedialog

    if not file_path:
        print('No file path given. Choose path.')
        root = tk.Tk()
        file_path = filedialog.askopenfilename(title="Choose path")
        root.withdraw()  # hide main tkinter window
    else:
        print('Using the provided file path.')
        file_path=file_path

    return file_path

def OT_flexible_folder_reader(folder_path=None):
    if not folder_path:
        print('No file path given. Choose path.')
        root = tk.Tk()
        root.withdraw()  # hide main tkinter window
        folder_path = filedialog.askdirectory(title="Choose path")
    else:
        print('Using the provided file path.')
        folder_path=folder_path

    return folder_path

'''Misc'''
def find_filter_center(energy, function, flat_tolerance=1e-6, flat_width_threshold=3):
    max_val = np.max(function)
    plateau_indices = np.where(np.abs(function - max_val) < flat_tolerance)[0]

    if len(plateau_indices) >= flat_width_threshold:
        # Flat-top (Heaviside-like): take average of plateau region
        return np.mean(energy[plateau_indices])
    else:
        # Sharp peak (Gaussian-like): take single max index
        max_index = np.argmax(function)
        return energy[max_index]
def extract_numeric_suffix(filename):
    if filename.endswith(".ttbin") and not filename.endswith("-23.ttbin"):
        try:
            # Extract number between last dot and .ttbin
            return int(filename.rsplit('.', 2)[-2])
        except ValueError:
            return None
    return None

def stamps_channels_to_csv(timestamps, channels, load_path):
    combined_array=np.column_stack((timestamps, channels))

    save_path=load_path.replace('.ttbin', '.csv')

    print('Saving...')
    np.savetxt(save_path, combined_array, delimiter=',', header='Timestamp (ps), Channels')
    print('Done!\n\n')

import types
__all__ = [name for name, obj in globals().items()
           if not name.startswith('_') and isinstance(obj, types.FunctionType)]