# E4_offline
This package is for E4 offline stress level estimation, based on a training of WESAD data.

You need to put main.py, utils.py, GN_E4_stress.joblib and data folder under the same directory.


options:
--direc: the data directory to E4 recording
--window: segment length in second
--step: step length of sliding window, in second
--start: your session start time
--end: your session end time

Note: Your E4 recording should start before session starts and stop after session ends, otherwise valueError will be raised.

the input format of start and end can either be datetime or unix timestamp in UTC:

datetime format of Oct/20/2020 14:30:24:200 should be input as:
--start 2020 10 20 14 30 24 200 s

Here, 's' stands for summer timezone of Nashville: GMT/UTC-5
if it's winter timezone, simply replace 's' with 'w': GMT/UTC-6

UTC format of Nashville Time Oct/20/2020 14:30:24 should be like this: 1603222224

If your session started at Oct/8/2020 10:43:30:300 and your E4 recording was under directory '1602171700_A019C2', and you want to segment singal into 1 minute segments with 1 second as step length of sliding window:

To run the analysis, open a terminal:
$ python main.py --direc 1602171700_A019C2 --window 60 --step 1
                 --start 2020 10 8 10 43 30 300 s --end 2020 10 8 10 55 30 300 s

The prediction of stress level would be output in 'output_stress.csv' in the input direcotry.
