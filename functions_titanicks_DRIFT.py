import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
from pathlib import Path

from scipy.signal import convolve

def load_csv_to_numpy(exp_path):
    """
    Load the data from .csv files in exp_path.
    time_vect is found from *_data_export*.csv; frame_3d is obatined from the data in *_vsChem_export*.csv

    Parameters
    ----------
    exp_path : Path
        Path of the folder with .csv files

    Returns
    -------
    tuple
        time_vect : array specifying the time stamp for each frame
        frame_3d : 3d array of the pixels data

    """
    meta_filename = [i for i in exp_path.glob("*_data_export*.csv")]
    meta_filename = meta_filename[0] if len(meta_filename) > 0 else print(f'No meta csv file found in {exp_path}')
    df_meta = pd.read_csv(meta_filename)
    time_vect = df_meta['Time Elapsed'].dropna().to_numpy()
    # average_output = df_meta['Average Output'].dropna().to_numpy()

    data_filename = [i for i in exp_path.glob("*_vsChem_export*.csv")]
    data_filename = data_filename[0] if len(data_filename) > 0 else print(f'No data csv file found in {exp_path}')
    N, M, T = 56, 78, time_vect.shape[0]
    df_raw = pd.read_csv(data_filename, header=None).iloc[:, :N*M]
    frame_2d = df_raw.to_numpy()

    frame_3d = np.zeros((M, N, T))
    for i in range(T):
        frame_3d[:, :, i] = frame_2d[i, :].reshape(M, N, order='F')  # take row and reshape. append to 3d array

    return time_vect, frame_3d


def binary_file_read(file):
    """ Unpacks .bin file and returns list of the data in it

    Parameters
    ----------
    file : Path
        .bin file name
    Returns
    -------
    list
        data_list : list of the data in the input binary file
    """
    fw = open(file, 'rb')
    data_byte = fw.read()
    data_len = len(data_byte) >> 1
    data_list = []
    for n in range(0, data_len):
        (data, ) = struct.unpack('>H', data_byte[2*n:2*(n+1)])
        data_list.append(data)
    return data_list


def load_bin_data(exp_path, version):
    """ Unpacks the .bin file contained in the folder exp_path and returns list of the data

    Parameters
    ----------
    exp_path : Path
        path of the folder containing the .bin file
    version : str
        version of the chip, 'v2' or 'v4'

    Returns
    -------
    list
        data : list of the data from the binary file in the input folder
    """
    if version == 'v4':
        filename = [i for i in exp_path.glob("VF*.bin")]
    elif version == 'v2':
        filename = [i for i in exp_path.glob("*.bin")]
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented in load_bin_data')

    if len(filename) > 0:
        filename = filename[0]
    else:
        print(f'No bin file found in {exp_path}')
    data = binary_file_read(filename)
    return data


def binary_to_numpy(data, f_start, f_end, version):
    ''' Converts list of data from bin file to numpy representation

    Parameters
    ----------
    data : list
        list of input data to be converted into numpy array
    f_start : int
        first frame to be considered
    f_end : int
        last frame to be considered. To set it change code. Now it's len(data)//4372
    version : str
        version of the chip, 'v2' or 'v4'

    Returns
    -------
    tuple
        (time_vect, frame_3d, temp_vect) where
        - time_vect : np.array
            1D array containing the sampling times
        - frame_3d : np.array
            3D array containing time frames of the sensor array outputs. NxMxT, where NxM is the sensor array, T is time
        - temp_vect : np.array
            1D array containing some temperature information
    '''
    # convert data into 3D object [x, y, time]
    #f_end = len(data)//4372  # remove line when we move to GUI
    if version =='v2':
        N = 4372
    elif version == 'v4':
        N = 4374
    else:
        raise NotImplemented(f'The code for version {version} of the chip has not been implemented in binary_to_numpy')

    f_end = len(data)//N  # remove line when we move to GUI
    n_time = f_end - f_start + 1
    frame_3d = np.zeros((78, 56, n_time))
    time_vect = np.zeros(n_time)
    temp_vect = np.zeros(n_time)
    for i, n in enumerate(range(f_start - 1, f_end)):
        pack = data[n * N:(n + 1) * N]
        frame_1d = np.array(pack[:4368])
        frame_2d = frame_1d.reshape(78, 56, order='F')
        frame_3d[:, :, i] = frame_2d
        time_vect[i] = pack[4368] / 10
        temp_vect[i] = pack[4370]
    return time_vect, frame_3d, temp_vect


def split_chem_and_temp(arr):
    """ Separates temperature and chemical pixels.

    Parameters
    ----------
    arr : np.array
        3D array of chemical and temperature pixels (1 temp pixel for each 8 chem pixels)

    Returns
    -------
    tuple
        (arr_temp, arr_chem) where
        - arr_temp - 3D array of the temperature pixels
        - arr_chem - 3D array of the chemical pixels, where the temperature pixels are replaced by a chemical pixel whose value is the average of the ones surrounding it
    """
    arr_temp = arr[1::3, 1::3, :]  # obtain the temperature array by selecting one pixel every 3

    mask = np.ones((3, 3, 1)) / 8
    mask[1, 1, 0] = 0
    # mask = 1 1 1
    #        1 0 1
    #        1 1 1
    # 2D convolution of a signal with the mask above
    # results in an output signal where each value is the average of the surrounding ones in the input signal
    av_3d = convolve(arr, mask, mode='same')  # perform convolution

    arr_chem = arr.copy()  # copy the original to preserve the original chemical pixels

    arr_chem[1::3, 1::3, :] = av_3d[1::3, 1::3, :]  # substitute the temp pixels with the average found by convolution
    return arr_temp, arr_chem


def filter_by_vref(X, v_thresh=70):
    '''
    Identifies active pixels by checking if one of the first 10 derivatives d(i) is > v_thresh

    Parameters
    ---------
    X : np.array
        Input 2D array (T x NM). T = time samples, NM = total number of pixels
    v_thresh : int, optional
        Minimum value of the derivative d(i)=X(i+1)-X(i) in mV. Default is 70

    Returns
    -------
    np.array
        1D array of bool with dimension (NM). For each pixel, returns True if, during the first 10 samples,
        one of the derivatives is > v_thresh. The derivatives are calculated as d(i) = X(i+1)-X(i)
    '''
    return (np.diff(X[:10, :], axis=0) > v_thresh).any(axis=0)  # check if one of the first 10 derivatives is >v_thresh


def filter_by_vrange(X, v_range=(100, 900)):
    '''
    Identifies active pixels by checking that all the values are in v_range

    Parameters
    ---------
    X : np.array
        Input 2D array (T x NM). T = time samples, NM = total number of pixels
    v_range : (int, int), optional
        tuple containing the minimum and maximum allowable voltage in mV. Default is (100, 900)

    Returns
    -------
    np.array
        1D array of bool with dimension (NM). For each pixel, returns True if the value is always in v_range
    '''
    return (X < v_range[1]).all(axis=0) & (X > v_range[0]).all(axis=0)  # for each pixel, check if all the values are
    # within the given range


def time_to_index(times, time_vect):
    '''
    Returns index of the times closest to the desired ones time_vect

    Arguments
    ---------
    times : list
        list of integers containing the desired times
    time_vect : np.array
        array of the times at which the values are sampled

    Returns
    -------
    list
        for each element in the input list times, return an element in the output list
        with the index of the sample closest to the desired time
    '''
    indices = []
    for time in times:  # for each time in the input list
        indices.append( np.argmin(np.abs(time_vect - time)) )
        # find index of the sampled time (in time_vect) closest to the desired one (time)
    return indices


def load_and_preprocessing_DRIFT(exp_path, input_type='csv', version='v4', plt_summary=False, save_summary=False, print_status=False):
    ''' Load data from .bin/.csv file and preprocess it to find active pixels.

    Parameters
    ----------
    exp_path : Path
        path of the folder with the .bin/.csv fies
    input_type : string, optional
        'csv' or 'bin'. Other inputs will raise an error. Default 'csv'
    version : string, optional
        'v2' or 'v4' depending on how the data is stored. Default 'v4' (TODO specify better)
    plt_summary : bool, optional
        if plt_summary=True, a summary plot for the experiment is shown. Default False
    save_images : bool, optional
        if save_images=True, save the summary plot. Default False
    print_status : bool, optional
        if print_status=True, print status in terminal. Default False

    Returns
    -------
    tuple
        - time_vect_exp : time vector from the beginning of the experiment. Experiment is considered to start 10 samples
         after the highest disturbance in the data (corresponding to sample insertion)
        - arr_chem_bs_mean : chemical data for the experiment. This is background-subtracted average data from the active pixels
    '''

    # LOAD .BIN/.CSV DATA
    if print_status:
        print('Load data...')
    if input_type == 'bin':
        data = load_bin_data(exp_path, version)
        time_vect, frame_3d, temp_vect = binary_to_numpy(data, 1, 976, version)
    elif input_type == 'csv':
        time_vect, frame_3d = load_csv_to_numpy(exp_path)
    else:
        raise NotImplemented(f'The code to load data of input type {input_type} has not been implemented :(')

    # PREPROCESSING
    if print_status:
        print(f'Data loaded. \nPreprocessing start...')

    # Split temperature and chemical data - and obtain 2D array
    arr_temp, arr_chem_3d = split_chem_and_temp(frame_3d)
    n_time = arr_chem_3d.shape[2]
    arr_chem_2d = frame_3d.reshape(-1, n_time, order='C').T  # reshape the 3D matrix N x M x T to a 2D matrix of dimensions T x NM

    # Filter active pixels by Vref and Vrange
    find_active_pixels = lambda x: filter_by_vref(x, v_thresh=50) & filter_by_vrange(x, v_range=(50, 950))
    idx_active = find_active_pixels(arr_chem_2d)

    # Find background-subtracted average data
    arr_chem_bs_mean = np.mean(arr_chem_2d[:, idx_active] - arr_chem_2d[0, idx_active], axis=1)
    if plt_summary:
        plot_exp_summary_DRIFT(exp_path, time_vect, arr_chem_3d, arr_chem_2d, idx_active, save_summary)

    if print_status:
        print('Preprocessing end.')
    return time_vect, arr_chem_bs_mean, arr_chem_2d[:, idx_active]


def plot_exp_summary_DRIFT(exp_path, time_vect, arr_chem_3d, arr_chem_2d, idx_active, save_plt):
    # PLOT
    fig, ax = plt.subplots(4, 1, figsize=(5, 12), dpi=100)
    fig.suptitle(f'{exp_path}')
    pos = ax[0].imshow(np.mean(arr_chem_3d, axis=2), cmap='cividis')
    fig.colorbar(pos, ax=ax[0])
    ax[0].set(title='Average chem data for each pixel')

    ax[1].plot(time_vect, arr_chem_2d)
    ax[1].plot(time_vect, np.mean(arr_chem_2d, axis=1), c='k', linewidth=3)
    ax[1].set(title='Chem data & average', xlabel='Time(s)', ylabel='mV')

    ax[2].imshow(idx_active.reshape(-1, 56), cmap='cividis')
    ax[2].set(title='Active pixels (filtered by Vref Vrange)')

    ax[3].plot(time_vect, arr_chem_2d[:, idx_active] - arr_chem_2d[0, idx_active])
    ax[3].plot(time_vect, np.mean(arr_chem_2d[:, idx_active] - arr_chem_2d[0, idx_active], axis=1), c='k', linewidth=3)
    ax[3].set(title='BS Chem data from active pixels', xlabel='Time(s)', ylabel='mV')

    plt.tight_layout()
    if save_plt:
        plt_path = Path(exp_path, 'exp_summary.png')
        plt.savefig(plt_path)
        print(f'Saved {plt_path}')
    plt.show()
    return
