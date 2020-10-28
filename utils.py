#!/usr/bin/env python
""" Data reader and feature extracter modules for E4 offline processing

__Author__='Guangtao Nie'
__Institution__='RASL Lab, Vanderbilt Univ'
__version__='0.1'
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import scipy
import heartpy as hp
import datetime
from datetime import timezone
from biosppy.signals.eda import kbk_scr
import os
from datetime import datetime


class Reader():
    def __init__(self, freqs, step_ratio, window):
        self.acc_freq = freqs['acc']
        self.bvp_freq = freqs['bvp']
        self.eda_freq = freqs['eda']
        self.temp_freq = freqs['temp']
        self.step_ratio = step_ratio
        self.window = window

    def date_converter(self, year, month, day, hour, minute, second, msec, season):
        if season.startswith('s'):
            lag = 5
        elif season.startswith('w'):
            lag = 6
        else:
            raise ValueError(
                'season should be input, either winter/w or summer/s for timezone calculation.')
        try:
            dt = datetime(year, month, day, hour, minute, second, msec)
            # converted to GMT-5, Nashville local time zone
            timestamp = dt.replace(tzinfo=timezone.utc).timestamp()+lag*60*60
            return timestamp
        except:
            raise ValueError('input datetime not valid!')

    def get_data(self, direc, start, end):
        acc = pd.read_csv(os.path.join(direc, 'ACC.csv'),
                          header=None).to_numpy()
        baseline = acc[0][0]
        print('E4 recording start at:', baseline)
        if start < baseline:
            raise ValueError('session started before E4 recording!')
        if (end-baseline)*self.acc_freq > acc.shape[0]-2:
            raise ValueError('E4 recording finished before session end!')
        start, end = int(start-baseline), int(end-baseline)
        # add 2 because frst 2 rows in a file are timestamp and freq
        acc = acc[int(start*self.acc_freq)+2:int(end*self.acc_freq)+2]
        bvp = pd.read_csv(os.path.join(direc, 'BVP.csv'),
                          header=None).to_numpy()
        bvp = bvp[int(start*self.bvp_freq)+2:int(end*self.bvp_freq)+2]
        eda = pd.read_csv(os.path.join(direc, 'EDA.csv'),
                          header=None).to_numpy()
        eda = eda[int(start*self.eda_freq)+2:int(end*self.eda_freq)+2]
        temp = pd.read_csv(os.path.join(direc, 'TEMP.csv'),
                           header=None).to_numpy()
        temp = temp[int(start*self.temp_freq)+2:int(end*self.temp_freq)+2]
        ACC_data, BVP_data, EDA_data, TEMP_data = [], [], [], []
        for i in np.arange(0, end-start-self.window, int(self.step_ratio*self.window)):
            ACC_data.append(
                acc[i*self.acc_freq:(i+self.window)*self.acc_freq])
            BVP_data.append(
                bvp[i*self.bvp_freq:(i+self.window)*self.bvp_freq])
            EDA_data.append(
                eda[i*self.eda_freq:(i+self.window)*self.eda_freq])
            TEMP_data.append(
                temp[i*self.temp_freq:(i+self.window)*self.temp_freq])
        return np.asarray(ACC_data), np.asarray(BVP_data), np.asarray(EDA_data), np.asarray(TEMP_data)


class Extract_Features():
    def __init__(self, freqs):
        self.acc_freq = freqs['acc']
        self.bvp_freq = freqs['bvp']
        self.eda_freq = freqs['eda']
        self.temp_freq = freqs['temp']

    def acc_features(self, acc):
        # Mu, SD, Integral, Peaks = [], [], [], []
        # features = []
        # for acc in ACC:  # ACC shape (3,), acc shape (,320,3)
        M = np.zeros((acc.shape[0], 15))
        # print('acc shape:{}'.format(acc.shape))
        M[:, 0:3] = acc.mean(axis=1)
        # print('mu shape:{}'.format(mu.shape))
        M[:, 3:6] = acc.std(axis=1)
        # print('std shape:{}'.format(sd.shape))
        M[:, 6:9] = M[:, 0:3]*acc.shape[1]
        num_peaks = np.zeros((acc.shape[0], 3))
        for i in np.arange(acc.shape[0]):
            for j in np.arange(3):
                p, _ = find_peaks(
                    acc[i, :, j], distance=0.25*self.acc_freq)
                num_peaks[i, j] = p.size
        M[:, 9:12] = num_peaks
        M[:, 12] = M[:, 0:3].sum(axis=1)  # summed mu
        M[:, 13] = M[:, 3:6].sum(axis=1)  # summed sd
        M[:, 14] = M[:, 6:9].sum(axis=1)  # summed integral
        return M

    def bvp_features(self, bvp):
        M = np.zeros((bvp.shape[0], 16))
        for i in np.arange(bvp.shape[0]):
            try:
                # calc_freq=True,freq_method='fft',interp_clipping=True)
                _, m = hp.process(bvp[i], self.bvp_freq)
                M[i] = np.asarray(list(m.values()))
            except:
                M[i] = np.zeros(16)
        return M

    def eda_features(self, eda_):
        M = np.zeros((eda_.shape[0], 12))
        eda_ = np.squeeze(eda_, axis=2)
        for i in np.arange(eda_.shape[0]):
            # _,filtered,onsets,peaks,amplitudes=eda(eda_[i],eda_freq)
            # onsets,peaks,amplitudes=basic_scr(eda_[i],eda_freq)
            mean_eda = eda_[i].mean()
            sd_eda = eda_[i].std()
            minx = eda_[i].min()
            maxx = eda_[i].max()
            slope, _, _, _, _ = scipy.stats.linregress(
                np.arange(eda_[i].size), eda_[i])
            rang = maxx-minx
            while (eda_[i] < 0).any():
                eda_[i][eda_[i] < 0] = 0
            try:
                onsets, peaks, amplitudes = kbk_scr(eda_[i], self.eda_freq)
                num_scr = peaks.size
                for peak, amp in zip(peaks, amplitudes):
                    scl = eda_[peak]-amp
                mean_scl = scl.mean()
                sd_scl = scl.std()
                mean_scr = amplitudes.mean()
                sd_scr = amplitudes.std()
                corr_scl = scipy.stats.pearsonr(
                    np.arange(eda_[i].size), eda_[i])[0]
            except:
                print('{}th segment in eda of length:{} failed to extract eda features and all set to be 0'.format(
                    i, eda_[i].size))
                num_scr, mean_scl, sd_scl, mean_scr, sd_scr, corr_scl = 0, 0, 0, 0, 0, 0
            M[i] = np.asarray([mean_eda, sd_eda, minx, maxx, slope, rang,
                               num_scr, mean_scl, sd_scl, mean_scr, sd_scr, corr_scl])
        return M

    def temp_features(self, temp):
        M = np.zeros((temp.shape[0], 6))
        temp = np.squeeze(temp, axis=2)
        for i in np.arange(temp.shape[0]):
            mean_t = temp[i].mean()
            sd_t = temp[i].std()
            min_t = temp[i].min()
            max_t = temp[i].max()
            rang = max_t-min_t
            slope, _, _, _, _ = scipy.stats.linregress(
                np.arange(temp[i].size), temp[i])
            M[i] = np.asarray([mean_t, sd_t, min_t, max_t, rang, slope])
        return M

    def eda_bvp_nan(self, eda_features):
        for i in np.arange(eda_features.shape[0]):
            k = np.argwhere(np.isnan(eda_features[i]))
            eda_features[i][k] = 0
        for eda in eda_features:
            if np.isnan(eda).any():
                raise ValueError('nan still exists!')
