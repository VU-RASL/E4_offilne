#!/usr/bin/env python
""" This is for E4 offline processing
The input data should be in the standard format of Empatica E4.

__Author__='Guangtao Nie'
__Institution__='RASL Lab, Vanderbilt Univ'
__version__='0.1'
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import argparse
from utils import Reader, Extract_Features
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


freqs = {'acc': 32, 'bvp': 64, 'eda': 4, 'temp': 4}
labels_annotation = {0: 'low', 1: 'high'}


def main(args):
    clf = joblib.load('GN_E4_stress.joblib')

    print(args.start,args.end)
    step_ratio=args.step/args.window
    reader = Reader(freqs, step_ratio, args.window)
    if len(args.start) > 1:
        start=[int(x) for x in args.start if x.isnumeric()]
        start.append(args.start[-1])
        start = reader.date_converter(*start)
    else:
        start = float(args.start[0])
    if len(args.end) > 1:
        end=[int(x) for x in args.end if x.isnumeric()]
        end.append(args.end[-1])
        end = reader.date_converter(*end)
    else:
        end = float(args.end[0])
    print('session start time, end time:', start, end)
    acc, bvp, eda, temp = reader.get_data(args.direc, start, end)

    extractor = Extract_Features(freqs)
    acc_features = extractor.acc_features(acc)
    bvp_features = extractor.bvp_features(bvp)
    eda_features = extractor.eda_features(eda)
    temp_features = extractor.temp_features(temp)
    extractor.eda_bvp_nan(eda_features)
    extractor.eda_bvp_nan(bvp_features)
    print('shapes:', acc.shape, bvp.shape,
          eda.shape, temp.shape)
    print('feature shapes:', acc_features.shape,
          bvp_features.shape, eda_features.shape, temp_features.shape)

    features = np.hstack([acc_features, bvp_features,
                          eda_features, temp_features])
    D_test = xgb.DMatrix(features)
    preds = clf.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    labels = np.asarray([labels_annotation[i]
                         for i in best_preds]).reshape((-1, 1))
    best_preds = best_preds.reshape((-1, 1))
    print('preds shape:', best_preds.shape, labels.shape)

    timestamp=np.arange(int(start)+args.window,int(end),args.step).reshape((-1,1))
    #print(timestamp.shape)
    df = pd.DataFrame(np.hstack([timestamp,best_preds, labels]), columns=[
                      'timestamp','output', 'annotation'], index=None)
    df.to_csv(os.path.join(args.direc, 'output_stress.csv'), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direc", help="directory of single session", type=str, default='1602171700_A019C2')
    parser.add_argument(
        "--window", help="input signal length in seconds", type=int, default=60)
    parser.add_argument(
        "--start", help="session start time (can either be datetime or unix timestamp in UTC)", nargs='+', default=[2020, 10, 8, 10, 43, 30, 300, 's'])
    parser.add_argument(
        "--end", help="session end time (can either be datetime or unix timestamp in UTC)", nargs='+', default=[2020, 10, 8, 10, 55, 30, 300, 's'])
    parser.add_argument(
        "--step", help="step length relative to window length", type=int, default=1)
    args = parser.parse_args()
    argv = args.__dict__
    main(args)
