import pickle

import numpy as np
import pandas as pd

from modeling.feature_engineering import FeatureEngineering
from plots.plots import plot_one_cycle, plot_capacity, plot_sections


class DataPreprocess:
    """
    This class downloads the data, converts the downloaded mat file to pandas dataframe and resmaple it so that all the
    data are sampled at the same time span.

    :param final_file: The final pandas is saved to a pickle for further use.
    :param args: The global variables tha are passed around in the prohect.
    :param final_file_path: The path for the saved pickle file.
    """
    def __init__(self, args, plot_inputs=False, plot_features_flag=False,plot_ser='b1c0'):
        self.serial_no_list = None
        self.out_main_df = None
        self.cyc_df_resampled = None
        self.final_df = None
        self.input_columns = None
        self.args = args

        # Extracting the time from resample time string.
        self.timestep = int(args.resample_time.replace('s', ''))

        self.plot_inputs_flag = plot_inputs
        self.plot_ser = plot_ser
        self.plot_features_flag = plot_features_flag

    def load_data(self):
        self.cyc_df_resampled = pickle.load(open(self.args.main_path + 'severson_resampled.pkl', 'rb'))
        self.out_main_df = pickle.load(open(self.args.main_path + 'capacity_df.pkl', 'rb'))
        plot_capacity(self.out_main_df, self.args)

    def add_slopes(self, feature='Voltage(V)'):
        # Adding voltage and current slope
        self.cyc_df_resampled[f'{feature}_slope'] = self.cyc_df_resampled[f'{feature}'].shift(1)
        self.cyc_df_resampled[f'{feature}_slope'] = self.cyc_df_resampled[f'{feature}_slope'].fillna(method='bfill')
        self.cyc_df_resampled[f'{feature}_slope'] = self.cyc_df_resampled[f'{feature}'] - self.cyc_df_resampled[
            f'{feature}_slope']
        self.cyc_df_resampled[f'{feature}_slope_rounded'] = self.cyc_df_resampled[f'{feature}_slope'].astype(
            float).round(1)

    def add_features(self):
        # *******************************************************************
        # *********************** FEATURE CALCULATION ***********************
        # *******************************************************************
        """
        Sections:
        Section I: Charging region, Constant-current part 1

        Section II: Charging region, Constant-current part 2

        Section III: Charging region, Constant-voltage

        Section IV: Discharging region

        Section V: Rest region
        """

        phys_list = ['Voltage(V)']

        # The following class will create the features for each section.
        feature_maker = FeatureEngineering(self.args, input=self.cyc_df_resampled)

        input_df = feature_maker.input

        # Plotting one cycle versus cycle number.
        plot_one_cycle(input_df, self.args, serial_no='b1c0', Cycle_Index=10)

        # Conditions for separating the sections
        conds = {}

        conds['I'] = (feature_maker.input['Current(A)'] > 2) & (
                abs(feature_maker.input['Current(A)_slope'].astype(float).round(1)) < 1e-1) & \
                     (abs(feature_maker.input['Voltage(V)_slope'].astype(float).round(1)) > 1e-1)

        conds['II'] = (abs(feature_maker.input['Current(A)'] < 2)) & (
                abs(feature_maker.input['Current(A)_slope'].astype(float).round(1)) < 1e-1) & \
                      (abs(feature_maker.input['Voltage(V)_slope'].astype(float).round(1)) > 1e-1)

        conds['III'] = (feature_maker.input['Voltage(V)'] > 2) & (
                abs(feature_maker.input['Voltage(V)_slope'].astype(float).round(1)) < 1e-1)

        conds['IV'] = (abs(feature_maker.input['Current(A)'] < -2)) & (
                abs(feature_maker.input['Current(A)_slope'].astype(float).round(1)) < 1e-1) & \
                      (abs(feature_maker.input['Voltage(V)_slope'].astype(float).round(1)) > 1e-1)

        conds['V'] = (feature_maker.input['Voltage(V)'] < 2.5) & (
                abs(feature_maker.input['Voltage(V)_slope'].astype(float).round(1)) < 1e-1)

        for phys in phys_list:
            for key, value in conds.items():
                feature_maker.calculate_features(phys_var=phys, section=key, cond=value)

        if self.plot_features_flag:
            # Plotting the features.
            plot_sections(feature_maker.input, self.args, conds, self.plot_ser)

        final_df = feature_maker.final_step()

        input_columns = list(final_df.columns)
        self.input_columns = list(set(input_columns) - {'serial_no', 'capacity_flag'})
        self.final_df = pd.merge(final_df,
                                 self.out_main_df[['serial_no', 'Cycle_Index', 'Capacity', 'datetime_c2_max']],
                                 on=['serial_no', 'Cycle_Index'], how='inner')

        self.serial_no_list = final_df['serial_no'].unique()

    def prepare(self):
        self.load_data()

        # Adding the slopes of the voltage and current for section creation and feature engineering.
        self.add_slopes(feature='Voltage(V)')
        self.add_slopes(feature='Current(A)')

        self.add_features()
        self.final_df = self.final_df.sort_values(['serial_no', 'datetime_c2_max'])
        final_cols = self.input_columns + ['serial_no', 'datetime_c2_max', 'Capacity']
        return self.final_df[final_cols]
