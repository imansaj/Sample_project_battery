import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import requests as requests
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle

from tqdm import tqdm
from mat4py import loadmat

from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import urllib.request

def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    # Can also replace 'file' with a io.BytesIO object
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

def data_maker(args):
    filename = '2018-04-12_batchdata_updated_struct_errorcorrect.mat'
    my_file = Path(args.main_path + filename)
    if not my_file.is_file():
        print('Downloading the mat file.')
        download("https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download", args.main_path + filename)

    print('Starting converting mat file to batch file.')
    matFilename = args.main_path + filename
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

    print('Finished converting mat file to batch file.')


def pandas_maker(args):
    print('Starting creating pandas dataframe from mat file.')

    batch1 = pickle.load(open(args.main_path + 'batch1.pkl', 'rb'))

    def main_data(battery, cycle):
        temp_df = pd.DataFrame()

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

        temp_df2 = pd.DataFrame()

        temp_df2['Qdlin'] = batch1[battery]['cycles'][cycle]['Qdlin']
        temp_df2['Tdlin'] = batch1[battery]['cycles'][cycle]['Tdlin']
        temp_df2['dQdV'] = batch1[battery]['cycles'][cycle]['dQdV']

        temp_df2['cycle'] = int(cycle)
        temp_df2['battery_name'] = battery
        temp_df2['charge_policy'] = batch1[battery]['charge_policy']
        temp_df2['cycle_life'] = np.squeeze(batch1[battery]['cycle_life'])
        temp_df2['counter'] = temp_df2.index

        return temp_df, temp_df2

    list_main = [main_data(battery, cycle) for battery in batch1.keys() for cycle in batch1[battery]['cycles']]

    del batch1
    main_df = pd.concat([x[0] for x in list_main])
    main_df2 = pd.concat([x[1] for x in list_main])

    main_df.to_pickle(args.main_path + 'severson_main.pkl')
    main_df2.to_pickle(args.main_path + 'severson_main_dqlin.pkl')
    print('Finished creating pandas dataframe from mat file.')

