# -*- coding: utf-8 -*-
"""

@author: Iman Sajedian
"""

from functools import reduce

# %%
import pandas as pd


# %%
# from plots.plots import manual_vs_real_temp


class FeatureEngineering:
    def __init__(self, args, input=None, output=None):
        """
        This class calculates the features for machine learning model.
        There are five sections for each cycle:
        1.	I, I = const.: Constant current (CC)
        2.	II, V = const.: Constant Voltage(CV)
        3.	III, I = 0: Rest (after charge)(OCV_III)
        4.	IV: Discharge
        5.	V, I = 0: Rest (after discharge)(OCV_V)
        """
        self.args = args
        self.input = input
        self.output = output
        self.input['Voltage(V)_slope'] = self.input['Voltage(V)_slope'] * 1000

        self.data_frames = []
        self.binary_dict_list = []

    def calculate_features(self, phys_var='Voltage(V)', section='I', cond=None):
        """
               This function calculates parameters of Constant Current part of the charge curve like max, sum,
               ... of Voltage or current and their duration and adds them to the final features.

               Parameters
               ----------
               phys_var : str, optional
                   This can be either Voltage(V) or Current(A)
                   :param phys_var:
                   :param cond:
                   :param section:
               """
        temp_list = ['', '_slope']
        for type_ in temp_list:
            temp_str = 'cc_charge_' + phys_var + '_I_'
            temp_str = phys_var + f'_{section}' + type_

            d_names = {'sum': temp_str + '_sum', 'count': temp_str + '_duration', 'mean': temp_str + '_mean',
                       'min': temp_str + '_min', 'max': temp_str + '_max', 'std': temp_str + '_std',
                       'skew': temp_str + '_skew', 'kurt': temp_str + '_kurt'}

            df_ = self.input.loc[cond].groupby(['serial_no', 'Cycle_Index']). \
                agg({phys_var + type_: ['sum', 'count', 'mean', 'skew', 'std', pd.DataFrame.kurt]}).rename(
                columns=d_names)

            df_.columns = df_.columns.droplevel(0)
            df_ = df_.reset_index()
            df_ = df_.fillna(0)
            df_[temp_str + '_ub'] = df_[temp_str + '_mean'] + df_[temp_str + '_std']
            df_[temp_str + '_lb'] = df_[temp_str + '_mean'] - df_[temp_str + '_std']
            self.data_frames.append(df_)

    def final_step(self):
        """
        This function merges all the calculated features in the class and return them to the user.

        """

        def nan_filler(df):
            df = df.interpolate(method='linear', limit_direction='forward', axis=0)
            df = df.fillna(method='bfill').fillna(method='ffill')
            return df

        main_df = reduce(lambda left, right: pd.merge(left, right, on=['serial_no', 'Cycle_Index'],
                                                      how='outer'), self.data_frames)

        main_df = main_df.groupby(['serial_no']).apply(nan_filler).reset_index(drop=True)

        return main_df
