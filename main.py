"""
State Of Health of battery cells estimation using machine learning.

@author: Iman Sajedian

This project tries to predict the capacity of battery cells, using their voltage obtained from each cycle. More details
is given on the github page:
https://github.com/imansaj/Sample_project_battery

The data information can be found here:
https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

"""


import argparse
import os
import sys
import time
from datetime import timedelta

from Data.read_data import DataPrepare
from modeling.feature_extraction import DataPreprocess
from modeling.models import MainModel
from plots.plots import plot_features, plot_final_results

parser = argparse.ArgumentParser()

parser.add_argument('--resample_time', type=str, default='10s')

# Main paths. Please change it as you desire.
parser.add_argument('--main_path', type=str, default='D:/Severson_battery_data/')

parser.add_argument('--random_state', type=int, default=10)
parser.add_argument('--num_cells', type=int, default=12, help='The number of cells used for training. There are total '
                                                              'of 46 cells in the dataset.')

args = parser.parse_args()

if __name__ == '__main__':

    t0 = time.time()

    args.plot_path = args.main_path + 'plots/'
    if not os.path.exists(args.main_path):
        os.makedirs(args.main_path)

    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    # The following class downloads the data and resample it for later use.
    initial_data = DataPrepare(args)
    initial_data.prepare()

    # The following class does the feature engineering and prepare the data for  modelling.
    data_processed = DataPreprocess(args, plot_inputs=False, plot_features_flag=True, plot_ser='b1c0')
    main_df = data_processed.prepare()

    plot_features(main_df, args, plot_ser='b1c0')

    # The following class trains the model on the Kfold data.
    model1 = MainModel(args, no_features=37)
    final_df_ = model1.train_cv(main_df, splits=5, plot_flag=False, plot_importance=True)

    final_df_ = final_df_.sort_values(by=['serial_no', 'Cycle_Index'])
    plot_final_results(final_df_, args)

    t1 = time.time()
    print(f'Total elapsed time:{timedelta(seconds=t1 - t0)}.')
    print('Finished!')

    sys.exit()
