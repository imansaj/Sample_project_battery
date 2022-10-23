import pickle

import numpy as np
import pandas as pd

from modeling.feature_engineering import FeatureEngineering
from plots.plots import plot_one_cycle, plot_capacity, plot_sections


class DataPreprocess:
    def __init__(self, args, pack=1, hours=30, plot_inputs=False, plot_features_flag=False,
                 plot_ser=125):
        self.serial_no_list = None
        self.out_main_df = None
        self.cyc_df_resampled = None
        self.final_df = None
        self.input_columns = None
        self.args = args
        self.random_state = 8
        self.timestep = int(args.resample_time.replace('s', ''))
        self.hours = hours
        self.plot_inputs_flag = plot_inputs
        self.pack_no = pack
        self.plot_ser = plot_ser
        self.plot_features_flag = plot_features_flag

    def load_data(self):
        self.cyc_df_resampled = pickle.load(open(self.args.main_path + 'severson_resampled.pkl', 'rb'))
        self.out_main_df = pickle.load(open(self.args.main_path + 'capacity_df.pkl', 'rb'))
        plot_capacity(self.out_main_df, self.args)

    def remove_outliers(self):
        def is_outlier(s, multiplier=2):
            lower_limit = s.mean() - (s.std() * multiplier)
            upper_limit = s.mean() + (s.std() * multiplier)
            return ~s.between(lower_limit, upper_limit)

        self.out_main_df.loc[self.out_main_df.groupby('serial_no')[self.args.output_feature].apply(
            is_outlier), self.args.output_feature] = np.nan
        self.out_main_df[self.args.output_feature] = self.out_main_df.groupby('serial_no')[
            self.args.output_feature].transform(
            lambda v: v.bfill().ffill())

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
        I: Constant current (Charging)
        II: Constant voltage (Charging)
        III: Rest after charging
        IV: Discharge
        V: Rest After discharge
        """

        phys_list = ['Voltage(V)']

        feature_maker = FeatureEngineering(self.args, input=self.cyc_df_resampled)

        input_df = feature_maker.input

        self.add_slopes(feature='Temperature(C)')

        plot_one_cycle(input_df, self.args, serial_no='b1c0', Cycle_Index=10)

        # Conditions for different features
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
        self.add_slopes(feature='Voltage(V)')
        self.add_slopes(feature='Current(A)')
        self.add_features()
        self.final_df = self.final_df.sort_values(['serial_no', 'datetime_c2_max'])
        final_cols = self.input_columns + ['serial_no', 'datetime_c2_max', 'Capacity']
        return self.final_df[final_cols]
