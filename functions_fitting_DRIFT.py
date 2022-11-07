import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

from functions_titanicks_DRIFT import time_to_index


def split_2_segments(X, time_vect, min_split=10, show_plt=False, save_plt=False, exp_path=None):
    # Find Boundaries between experiment segments
    min_time_idx, max_time_idx = time_to_index([min_split*60, min_split*60+60], time_vect)
    dist1_idx = np.argmax(np.mean(X, axis=1)[min_time_idx:max_time_idx]) + min_time_idx

    # NORMALISE
    time_vect = time_vect.reshape(-1, 1)
    x_scaler = MinMaxScaler()
    time_vect = x_scaler.fit_transform(time_vect).reshape(-1)
    y_scaler = MinMaxScaler()
    X = y_scaler.fit_transform(X)

    # SPLIT SEGMENTS
    t1 = time_vect[50:dist1_idx-20]
    x1 = X[50:dist1_idx-20, :]
    t2 = time_vect[dist1_idx+50:]
    x2 = X[dist1_idx+50:, :]

    # TODO identify the end in a clever way

    if show_plt:
        fig, ax = plt.subplots(1, 1, dpi=200)
        ax.plot(time_vect, X)
        ax.plot(time_vect, np.mean(X, axis=1), c='k', linewidth=3)

        ax.axvline(x=time_vect[dist1_idx])

        ax.axvspan(time_vect[50], time_vect[dist1_idx-20], alpha=0.5, color='y')
        ax.axvspan(time_vect[dist1_idx+50], time_vect[-1], alpha=0.5, color='y')
        if save_plt and exp_path == None:
            print('ERROR. Could not save the segments plot because the path is not given.')
        elif save_plt:
            plt_path = Path(exp_path, 'segments_summary.png')
            plt.savefig(plt_path)
            print(f'Saved {plt_path}')
        plt.show()

    return t1, x1, t2, x2, dist1_idx


def split_3_segments(X, time_vect, show_plt=False, save_plt=False, exp_path=None):
    # Find Boundaries between experiment segments
    min_time_idx, max_time_idx = time_to_index([10*60, 10*60+60], time_vect)
    dist1_idx = np.argmax(np.mean(X, axis=1)[min_time_idx:max_time_idx]) + min_time_idx

    min_time_idx, max_time_idx = time_to_index([20*60, 20*60+60], time_vect)
    dist2_idx = np.argmax(np.mean(X, axis=1)[min_time_idx:max_time_idx]) + min_time_idx

    # NORMALISE
    time_vect = time_vect.reshape(-1, 1)
    x_scaler = MinMaxScaler()
    time_vect = x_scaler.fit_transform(time_vect).reshape(-1)
    y_scaler = MinMaxScaler()
    X = y_scaler.fit_transform(X)

    # SPLIT SEGMENTS
    t1 = time_vect[50:dist1_idx-20]
    x1 = X[50:dist1_idx-20, :]
    t2 = time_vect[dist1_idx+50:dist2_idx-20]
    x2 = X[dist1_idx+50:dist2_idx-20, :]
    t3 = time_vect[dist2_idx+50:dist2_idx+450]
    x3 = X[dist2_idx+50:dist2_idx+450, :]

    # TODO identify the end in a clever way

    if show_plt:
        fig, ax = plt.subplots(1, 1, dpi=200)
        ax.plot(time_vect, X)
        ax.plot(time_vect, np.mean(X, axis=1), c='k', linewidth=3)

        ax.axvline(x=time_vect[dist1_idx])
        ax.axvline(x=time_vect[dist2_idx])

        ax.axvspan(time_vect[50], time_vect[dist1_idx-20], alpha=0.5, color='y')
        ax.axvspan(time_vect[dist1_idx+50], time_vect[dist2_idx-20], alpha=0.5, color='y')
        ax.axvspan(time_vect[dist2_idx+50], time_vect[dist2_idx+450], alpha=0.5, color='y')
        if save_plt and exp_path == None:
            print('ERROR. Could not save the segments plot because the path is not given.')
        elif save_plt:
            plt_path = Path(exp_path, 'segments_summary.png')
            plt.savefig(plt_path)
            print(f'Saved {plt_path}')
        plt.show()

    return t1, x1, t2, x2, t3, x3, dist1_idx, dist2_idx


def split_5_segments(X, time_vect, show_plt=False, save_plt=False, exp_path=None):
    # Find Boundaries between experiment segments
    min_time_idx, max_time_idx = time_to_index([10*60, 10*60+60], time_vect)
    dist1_idx = np.argmax(np.mean(X, axis=1)[min_time_idx:max_time_idx]) + min_time_idx

    min_time_idx, max_time_idx = time_to_index([20*60, 20*60+60], time_vect)
    dist2_idx = np.argmax(np.mean(X, axis=1)[min_time_idx:max_time_idx]) + min_time_idx

    min_time_idx, max_time_idx = time_to_index([30*60, 30*60+60], time_vect)
    dist3_idx = np.argmax(np.mean(X, axis=1)[min_time_idx:max_time_idx]) + min_time_idx

    min_time_idx, max_time_idx = time_to_index([40*60, 40*60+60], time_vect)
    dist4_idx = np.argmax(np.mean(X, axis=1)[min_time_idx:max_time_idx]) + min_time_idx

    # NORMALISE
    time_vect = time_vect.reshape(-1, 1)
    x_scaler = MinMaxScaler()
    #time_vect = x_scaler.fit_transform(time_vect).reshape(-1)
    time_vect = x_scaler.fit_transform(time_vect[:dist4_idx+500]).reshape(-1)
    y_scaler = MinMaxScaler()
    #X = y_scaler.fit_transform(X)
    X = y_scaler.fit_transform(X[:dist4_idx+500])

    # SPLIT SEGMENTS
    t1 = time_vect[50:dist1_idx-20]
    x1 = X[50:dist1_idx-20, :]
    t2 = time_vect[dist1_idx+50:dist2_idx-20]
    x2 = X[dist1_idx+50:dist2_idx-20, :]
    t3 = time_vect[dist2_idx+50:dist2_idx+450]
    x3 = X[dist2_idx+50:dist2_idx+450, :]

    t4 = time_vect[dist3_idx+50:dist4_idx-20]
    x4 = X[dist3_idx+50:dist4_idx-20, :]
    t5 = time_vect[dist4_idx+50:]
    #t5 = time_vect[dist4_idx+50:dist4_idx+500]
    x5 = X[dist4_idx+50:, :]
    #x5 = X[dist4_idx+50:dist4_idx+500, :]

    fig, ax = plt.subplots(1, 1, dpi=250)
    ax.plot(time_vect, X)
    ax.plot(time_vect, np.mean(X, axis=1), c='k', linewidth=3, label='average chem signal')
    #ax.plot(time_vect[:dist4_idx+500], X[:dist4_idx+500])
    #ax.plot(time_vect[:dist4_idx+500], np.mean(X[:dist4_idx+500], axis=1), c='k', linewidth=3, label='average chem signal')

    ax.axvline(x=time_vect[dist1_idx], ls='--',  color='b', label='pH changes')
    ax.axvline(x=time_vect[dist2_idx], ls='--',  color='b')
    ax.axvline(x=time_vect[dist3_idx], ls='--',  color='b')
    ax.axvline(x=time_vect[dist4_idx], ls='--',  color='b')

    ax.axvspan(time_vect[50], time_vect[dist1_idx-20], alpha=0.5, color=(255/255, 235/255, 42/255), label='pH9')
    ax.axvspan(time_vect[dist1_idx+50], time_vect[dist2_idx-20], alpha=0.5,  color=(255/255, 120/255, 0/255), label='pH8') # color=(255/255, 235/255, 42/255))
    ax.axvspan(time_vect[dist2_idx+50], time_vect[dist3_idx-20], alpha=0.5, color='r', label='pH7') #color=(255/255, 235/255, 42/255))
    ax.axvspan(time_vect[dist3_idx+50], time_vect[dist4_idx-20], alpha=0.5, color=(255/255, 120/255, 0/255)) #color=(255/255, 235/255, 42/255))
    ax.axvspan(time_vect[dist4_idx+50], time_vect[dist4_idx+500], alpha=0.5, color=(255/255, 235/255, 42/255))

    ax.grid(linestyle='--', color=(230/255, 230/255, 230/255))
    ax.legend(fontsize=13)
    ax.set_title('Chip 2. pH9-8-7-8-9', fontsize=18)
    ax.set_xlabel('Time (normalised)', fontsize=16)
    ax.set_ylabel('Vout (normalised)', fontsize=16)
    plt.savefig('ph98789_example.png')
    plt.show()

    return t1, x1, t2, x2, t3, x3, t4, x4, t5, x5


def split_5_segments_mean(time_vect, X, show_plt=False, save_plt=False, exp_path=None):
    # Find Boundaries between experiment segments
    min_time_idx, max_time_idx = time_to_index([10*60, 10*60+60], time_vect)
    dist1_idx = np.argmax(X[min_time_idx:max_time_idx]) + min_time_idx

    min_time_idx, max_time_idx = time_to_index([20*60, 20*60+60], time_vect)
    dist2_idx = np.argmax(X[min_time_idx:max_time_idx]) + min_time_idx

    min_time_idx, max_time_idx = time_to_index([30*60, 30*60+60], time_vect)
    dist3_idx = np.argmax(X[min_time_idx:max_time_idx]) + min_time_idx

    min_time_idx, max_time_idx = time_to_index([40*60, 40*60+60], time_vect)
    dist4_idx = np.argmax(X[min_time_idx:max_time_idx]) + min_time_idx

    # SPLIT SEGMENTS
    t1 = time_vect[50:dist1_idx-20]
    x1 = X[50:dist1_idx-20]
    t2 = time_vect[dist1_idx+50:dist2_idx-20]
    x2 = X[dist1_idx+50:dist2_idx-20]
    t3 = time_vect[dist2_idx+50:dist2_idx+450]
    x3 = X[dist2_idx+50:dist2_idx+450]

    t4 = time_vect[dist3_idx+50:dist4_idx-20]
    x4 = X[dist3_idx+50:dist4_idx-20]
    t5 = time_vect[dist4_idx+50:]
    x5 = X[dist4_idx+50:]

    return t1, x1, t2, x2, t3, x3, t4, x4, t5, x5


def decaying_exp(x, a, b):
    return a*(1-np.exp(-x/b))


def fit_pixels_interpolate_EXP2(time, X):
    """ Interpolates the curves for each pixel

    Parameters
    ----------
    time : ndarray
        times
    X : ndarray
        TxNM array to be interpolated
    idx_active : ndarray
        NM array specifying pixels that are active
    interpolate_idx : int
        interpolation is performed until this index

    Returns
    -------
    popt : ndarray
        optimal parameters for interpolation of each pixel, with shape 2xNM
    """
    popt = np.zeros((2, X.shape[1]))

    # for every pixel
    for i in range(X.shape[1]):
        # Fit the curve (interpolate)
        data = X[:, i]
        # popt[:, i], _ = curve_fit(decaying_exp, time, data_filt, p0=(-1, 10000))
        try:
            popt[:, i], _ = curve_fit(decaying_exp, time, data, p0=(-2, 0.5))
        except RuntimeError:
            print('RuntimeError in EXP2 fitting')
            popt[:, i] = 0.01

            #print(i, popt[:, i])
            #fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=50)
            #ax.plot(time, data)
            #ax.plot(time, decaying_exp(time, popt[0, i], popt[1, i]))
            #ax.set_title(f'{popt[:, i]}')
            #plt.show()
            popt[:, i] = 0

    return popt


def line(x, m, c):
    return m*x + c


def fit_pixels_interpolate_LIN(time, X):
    """ Interpolates the curves for each pixel

    Parameters
    ----------
    time : ndarray
        times
    X : ndarray
        TxNM array to be interpolated
    idx_active : ndarray
        NM array specifying pixels that are active
    interpolate_idx : int
        interpolation is performed until this index

    Returns
    -------
    popt : ndarray
        optimal parameters for interpolation of each pixel, with shape 2xNM
    """
    popt = np.zeros((2, X.shape[1]))

    # for every pixel
    for i in range(X.shape[1]):
        # Fit the curve (interpolate)
        data = X[:, i]
        # popt[:, i], _ = curve_fit(decaying_exp, time, data_filt, p0=(-1, 10000))
        try:
            popt[:, i], _ = curve_fit(line, time, data, p0=(-2, 0.1))
            # print(popt[:, i])
        except RuntimeError:
            print('RuntimeError in LIN fitting')
            popt[:, i] = 0.01

            # fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=50)
            # ax.plot(time, data)
            # ax.plot(time, line(time, popt[0, i], popt[1, i]))
            # ax.set_title(f'{popt[:, i]}')
            # plt.show()

    return popt

