# Data Link: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
# Download BEEP structured data: https://s3.amazonaws.com/publications.matr.io/1/final_data/FastCharge.zip
# https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download

import argparse
import sys

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
    args.data_path = 'C:/sim_data/2S_2P_01-Aug-2022'
    args.main_path = 'C:/sim_data'
    args.plot_path = args.main_path + '/plots'
    # data_maker(args)
    pandas_maker(args)
    if args.pack_type == '2S2P':
        args.file_name = 'data_2S2P.pkl'
    else:
        args.file_name = 'data_2P2S.pkl'



    sys.exit()
