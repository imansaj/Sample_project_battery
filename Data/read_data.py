import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mat4py import loadmat

from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import os
from sklearn.model_selection import train_test_split

def data_maker(args):

    matFilename = 'D:/Severson_battery_data/2018-04-12_batchdata_updated_struct_errorcorrect.mat'
    f = h5py.File(matFilename)
    batch = f['batch']

    num_cells = batch['summary'].shape[0]
    bat_dict = {}
    for i in range(num_cells):
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

    with open('D:/Severson_battery_data/batch1.pkl', 'wb') as fp:
        pickle.dump(bat_dict, fp)

    print('Finished data making for severeson.')

def pandas_maker(args):

        data_folder = 'D:/Severson_battery_data/'
        batch1 = pickle.load(open(r'D:/Severson_battery_data/batch1.pkl', 'rb'))

        main_df = pd.DataFrame()
        main_df2 = pd.DataFrame()

        for battery in batch1.keys():
            print(battery)
            main_df_temp = pd.DataFrame()
            main_df2_temp = pd.DataFrame()
            for cycle in batch1[battery]['cycles']:
                temp_df = pd.DataFrame()
                temp_df2 = pd.DataFrame()

                temp_df['I'] = batch1[battery]['cycles'][cycle]['I']
                temp_df['Qc'] = batch1[battery]['cycles'][cycle]['Qc']
                temp_df['Qd'] = batch1[battery]['cycles'][cycle]['Qd']
                temp_df['T'] = batch1[battery]['cycles'][cycle]['T']
                temp_df['V'] = batch1[battery]['cycles'][cycle]['V']
                temp_df['t'] = batch1[battery]['cycles'][cycle]['t']
                temp_df['counter'] = temp_df.index

                temp_df['cycle'] = int(cycle)
                temp_df['battery_name'] = battery
                temp_df['charge_policy'] = batch1[battery]['charge_policy']
                temp_df['cycle_life'] = np.squeeze(batch1[battery]['cycle_life'])

                temp_df2['Qdlin'] = batch1[battery]['cycles'][cycle]['Qdlin']
                temp_df2['Tdlin'] = batch1[battery]['cycles'][cycle]['Tdlin']
                temp_df2['dQdV'] = batch1[battery]['cycles'][cycle]['dQdV']

                temp_df2['cycle'] = int(cycle)
                temp_df2['battery_name'] = battery
                temp_df2['charge_policy'] = batch1[battery]['charge_policy']
                temp_df2['cycle_life'] = np.squeeze(batch1[battery]['cycle_life'])
                temp_df2['counter'] = temp_df2.index

                main_df_temp = pd.concat([main_df_temp, temp_df])
                main_df2_temp = pd.concat([main_df2_temp, temp_df2])

                main_df = pd.concat([main_df, temp_df])
                main_df2 = pd.concat([main_df2, temp_df2])

            main_df_temp.to_pickle(data_folder + 'temp/' + f'severson_{battery}.pkl')
            main_df2_temp.to_pickle(data_folder + 'temp/' + f'severson_{battery}_dqlin.pkl')

        main_df.to_pickle(data_folder + 'severson_main.pkl')
        main_df2.to_pickle(data_folder + 'severson_main_dqlin.pkl')
        print('Finished creating pandas dataframe from mat file.')

