# Data Link: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
# Download BEEP structured data: https://s3.amazonaws.com/publications.matr.io/1/final_data/FastCharge.zip
# https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from Data.read_data import data_maker, pandas_maker

parser = argparse.ArgumentParser()

# options: 'pack' or 'cell'
parser.add_argument('--test_mode', type=str, default='pack')

# options: '2S2P' or '2P2S'
parser.add_argument('--pack_type', type=str, default='2S2P')

# options: 'ML' or 'DL'
parser.add_argument('--model_type', type=str, default='ML')

parser.add_argument('--resample_time', type=str, default='1s')
parser.add_argument('--fault_start', type=int, default=1000)
parser.add_argument('--fault_stop', type=int, default=1010)

parser.add_argument('--read_data', dest='read_data', action='store_true')
parser.add_argument('--no_read_data', dest='read_data', action='store_false')
parser.set_defaults(read_data=False)

parser.add_argument('--DL', dest='DL', action='store_true')
parser.add_argument('--no_DL', dest='DL', action='store_false')
parser.set_defaults(DL=False)

args = parser.parse_args()

if __name__ == '__main__':
    args.main_path = 'D:/Severson_battery_data/'
    if not os.path.exists(args.main_path):
        os.makedirs(args.main_path)
    args.plot_path = args.main_path + '/plots'

    my_file = Path(args.main_path + 'batch1.pkl')
    if my_file.is_file():
        print('The batch file(batch1.pkl) already exists, no need to convert mat file to batch file!')
    else:
        data_maker(args)

    my_file = Path(args.main_path + 'severson_main.pkl')
    if my_file.is_file():
        print('The main file(severson_main.pkl) already exists, no need to create pandas dataframes from batch file!')
    else:
        pandas_maker(args)
    if args.pack_type == '2S2P':
        args.file_name = 'data_2S2P.pkl'
    else:
        args.file_name = 'data_2P2S.pkl'



    sys.exit()
