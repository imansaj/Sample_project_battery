import pickle


def preprocess(args):
    main_df = pickle.load(open(args.main_path + 'severson_main.pkl', 'rb'))
    main_df['cycle_battery'] = main_df['cycle'].astype(str) + main_df['battery_name']
    capacity_df_train = main_df[['Qd', 'cycle_battery', 'cycle', 'battery_name']].groupby(['cycle_battery']).agg(['max'])
    capacity_df_train.columns = capacity_df_train.columns.droplevel(1)

    a=2