import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests as requests
from tqdm import tqdm


class DataPrepare:
    """
    This class downloads the data, converts the downloaded mat file to pandas dataframe and resmaple it so that all the
    data are sampled at the same time span.

    :param final_file: The final pandas is saved to a pickle for further use.
    :param args: The global variables tha are passed around in the prohect.
    :param final_file_path: The path for the saved pickle file.
    """
    def __init__(self, args):
        self.final_file = None
        self.args = args
        self.final_file_path = Path(self.args.main_path + 'severson_resampled.pkl')

    def prepare(self):

        if self.final_file_path.is_file():
            print('The resampled file(severson_resampled.pkl) already exists, no need to resample the main dataframe!')
        else:
            self.data_maker()
            self.pandas_maker()
            self.preprocess()

    def download(self, url: str, fname: str):
        """
        Downloading the data from cloud.
        :param url:
        :param fname:
        """
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
                desc=fname,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    def data_maker(self):
        """
        Downloads the mat file from the cloud and converts it to a dict.
        :return:
        """
        filename = '2018-04-12_batchdata_updated_struct_errorcorrect.mat'
        my_file = Path(self.args.main_path + filename)
        if not my_file.is_file():
            print('Downloading the mat file.')
            self.download("https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
                          self.args.main_path + filename)

        print('Starting converting mat file to dict.')
        matFilename = self.args.main_path + filename
        f = h5py.File(matFilename)
        batch = f['batch']

        num_cells = batch['summary'].shape[0]
        bat_dict = {}
        # 46

        for i in range(self.args.num_cells):
            cl = f[batch['cycle_life'][i, 0]][()]
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()
            summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
            summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
            summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
            summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
            summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
            summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
            summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
            summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
            summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                       'cycle': summary_CY}
            cycles = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            for j in range(cycles['I'].shape[0]):
                I = np.hstack((f[cycles['I'][j, 0]][()]))
                Qc = np.hstack((f[cycles['Qc'][j, 0]][()]))
                Qd = np.hstack((f[cycles['Qd'][j, 0]][()]))
                Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][()]))
                T = np.hstack((f[cycles['T'][j, 0]][()]))
                Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][()]))
                V = np.hstack((f[cycles['V'][j, 0]][()]))
                dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][()]))
                t = np.hstack((f[cycles['t'][j, 0]][()]))
                cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
                cycle_dict[str(j)] = cd

            cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
            key = 'b1c' + str(i)
            bat_dict[key] = cell_dict

        self.final_file = bat_dict
        # with open('D:/Severson_battery_data/self.final_file.pkl', 'wb') as fp:
        #     pickle.dump(bat_dict, fp)

        print('Finished converting mat file to dict.')

    def pandas_maker(self):
        print('Starting creating pandas dataframe from mat file.')

        # self.final_file = pickle.load(open(args.main_path + 'self.final_file.pkl', 'rb'))

        def main_data(battery, cycle):
            temp_df = pd.DataFrame()

            temp_df['I'] = self.final_file[battery]['cycles'][cycle]['I']
            temp_df['Qc'] = self.final_file[battery]['cycles'][cycle]['Qc']
            temp_df['Qd'] = self.final_file[battery]['cycles'][cycle]['Qd']
            temp_df['T'] = self.final_file[battery]['cycles'][cycle]['T']
            temp_df['V'] = self.final_file[battery]['cycles'][cycle]['V']
            temp_df['t'] = self.final_file[battery]['cycles'][cycle]['t']
            temp_df['counter'] = temp_df.index

            temp_df['cycle'] = int(cycle)
            temp_df['battery_name'] = battery
            temp_df['charge_policy'] = self.final_file[battery]['charge_policy']
            temp_df['cycle_life'] = np.squeeze(self.final_file[battery]['cycle_life'])

            return temp_df

        list_main = [main_data(battery, cycle) for battery in self.final_file.keys() for cycle in
                     self.final_file[battery]['cycles']]

        del self.final_file
        self.final_file = pd.concat(list_main)

        print('Finished creating pandas dataframe from mat file.')

    def time_diff(self, df_temp):
        df_temp = df_temp.assign(time_difference=(df_temp.t - df_temp.t.shift(1)))
        # Add value of 5 seconds between two cycles
        df_temp.time_difference = df_temp.time_difference.fillna(0.08)
        return df_temp

    def preprocess(self):
        print('Starting resampling pandas dataframe from main file.')

        self.final_file['cycle_battery'] = self.final_file['cycle'].astype(str) + self.final_file['battery_name']
        self.final_file.rename(
            columns={'battery_name': 'serial_no', 'I': 'Current(A)', 'V': 'Voltage(V)', 'T': 'Temperature(C)',
                     'cycle': 'Cycle_Index'}, inplace=True)

        # Unifying time by resampling.
        self.final_file = self.final_file.sort_values(by=['serial_no', 'Cycle_Index'])

        self.final_file = self.final_file.groupby(['serial_no', 'Cycle_Index']).apply(self.time_diff)

        cyc_df_resampled = self.final_file.groupby(['serial_no', 'Cycle_Index', 'charge_policy', 'cycle_battery'])[
            ['Current(A)', 'Qc', 'Qd', 'Temperature(C)', 'Voltage(V)', 't',
             'counter', 'cycle_life', 'time_difference']].apply(self.resample_time)

        cyc_df_resampled = cyc_df_resampled.reset_index()

        # Extracting capacity of each cell at the end of each cycle.
        capacity_df = cyc_df_resampled[['Qd', 'cycle_battery', 'Cycle_Index', 'serial_no', 'datetime_c2']].groupby(
            ['cycle_battery', ]).agg(['max'])
        capacity_df.columns = capacity_df.columns.droplevel(1)
        capacity_df.rename(columns={'Qd': 'Capacity', 'datetime_c2': 'datetime_c2_max'}, inplace=True)
        capacity_df.reset_index(inplace=True, drop=True)

        cyc_df_resampled.to_pickle(self.args.main_path + 'severson_resampled.pkl')
        capacity_df.to_pickle(self.args.main_path + 'capacity_df.pkl')
        print('Finished resampling main dataframe.')

    def resample_time(self, df):
        df['cum_dif'] = df['time_difference'].cumsum()
        df['beginning_date'] = datetime.datetime(year=2017, month=1, day=1, hour=0,
                                                 minute=0, second=0)
        df['time_added'] = pd.to_timedelta(df['cum_dif'], 'm')
        df['datetime_c2'] = df['beginning_date'] + df['time_added']
        df.index = df['datetime_c2']
        df = df[~df.index.duplicated(keep='first')]
        df = df.infer_objects()
        df = df.drop(['time_added', 'beginning_date', 'datetime_c2'], axis=1)

        if len(df) > 1:
            df = df.resample(self.args.resample_time).bfill(limit=1).interpolate()
            df.reset_index(inplace=True)
            return df
        else:
            return df
