import re
from pathlib import Path

import matplotlib
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold


def plot_capacity(df, args):
    """

    :param df:
    :param args:
    """
    font_text = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 14}
    fonts_main = {'family': 'Arial',
                  'weight': 'regular',
                  'size': 12}
    matplotlib.rc('font', **fonts_main)

    df = df.sort_values(by=['serial_no', 'Cycle_Index'])

    serial_no_list = df.serial_no.unique()

    plt.figure()
    for ser in serial_no_list:
        plt.plot(df.loc[df.serial_no == ser, 'Cycle_Index'], df.loc[df.serial_no == ser, 'Capacity'],
                 label=ser, linewidth=3.0)
    plt.xticks(**font_text)  # This argument will change the font.
    plt.yticks(**font_text)  # This argument will change the font.
    plt.legend()
    plt.ylabel(f'Capacity(Ah)', **font_text)
    plt.xlabel(f'Cycle', **font_text)
    plt.tight_layout()
    plt.savefig(args.plot_path + f'Capacity_vs_cycles.png')
    plt.show()
    plt.close()


def plot_one_cycle(df, args, serial_no='b1c0', Cycle_Index=1):
    temp = df[(df.Cycle_Index == Cycle_Index) & (df.serial_no == serial_no)]

    temp = temp.sort_values(by='datetime_c2')
    temp = temp.reset_index(drop=True)
    temp.reset_index(drop=True, inplace=True)
    time_step = int(re.findall(r'\d+', args.resample_time)[0])

    temp['Time(s)'] = temp.index * time_step
    temp['Time(m)'] = temp.index * time_step / 60
    temp['Time(h)'] = temp.index * time_step / 3600

    # **************** fig 3-a
    font_text = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 14}
    fonts_main = {'family': 'Arial',
                  'weight': 'regular',
                  'size': 12}
    matplotlib.rc('font', **fonts_main)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time(h)', **font_text)
    ax1.set_ylabel('Current(A)', color=color, **font_text)
    lns1 = ax1.plot(temp['Time(h)'], temp['Current(A)'], color=color, linewidth=3.0, label='Current(A)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Voltage(V)', color=color, **font_text)  # we already handled the x-label with ax1
    lns2 = ax2.plot(temp['Time(h)'], temp['Voltage(V)'], color=color, linewidth=3.0, label='Voltage(V)')
    ax2.tick_params(axis='y', labelcolor=color)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=3)

    plt.title(f'Cell({serial_no}) Cycle({Cycle_Index})', **font_text)

    fig.tight_layout()
    plt.savefig(args.plot_path + f'one_cycle.png', dpi=600)
    plt.show()
    plt.close()


def plot_sections_aux(df, args, ser, y_label='Voltage(V)', title='Section III', colormap='bwr', offset=False,
                      offset_value=0.05,
                      feat='Voltage(V)'):
    y = ['temp_index', 'Time(h)']
    x = df.columns

    cycles_list = [item for item in x if item not in y]
    NUM_COLORS = cycles_list[-1]

    font_text = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 14}
    fonts_main = {'family': 'Arial',
                  'weight': 'regular',
                  'size': 12}
    matplotlib.rc('font', **fonts_main)

    cm = plt.get_cmap(colormap)
    cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(cycles_list[i]) for i in range(len(cycles_list))])
    for i in range(len(cycles_list)):
        if offset == False:
            ax.plot(df.loc[1:, 'Time(h)'], df.loc[1:, cycles_list[i]], linewidth=3.0)
        else:
            ax.plot(df.loc[1:, 'Time(h)'], df.loc[1:, cycles_list[i]] + i * offset_value, linewidth=3.0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    plt.xticks(**fonts_main)  # This argument will change the font.
    plt.yticks(**fonts_main)  # This argument will change the font.

    plt.xlabel('Time(h)', **font_text)
    if offset == False:
        plt.ylabel(y_label, **font_text)
    else:
        plt.ylabel('Offsetted ' + y_label + ' values', **font_text)
    plt.title(f'Cell number ({ser}) ' + title, **font_text)
    cbar = plt.colorbar(scalarMap)
    cbar.ax.set_title('Cycles', **font_text)
    plt.tight_layout()
    plot_path = args.plot_path + 'sections/'
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_path + f'Sections_serial({ser})_{feat}_{title}.jpg', dpi=600)
    plt.show()

    plt.close()


def plot_sections_aux_2(df, cond, args, section='I', ser='b1c0', feat='Voltage(V)', slope=False, colormap='bwr',
                        offset=False,
                        offset_value=0.05):
    def my_fn(df_temp):
        df_temp['temp_index'] = range(len(df_temp))
        return df_temp

    if section == 'Full':
        temp1 = df[(df.serial_no == ser)]
    else:
        temp1 = df[(cond[section]) & (df.serial_no == ser)]
    temp2 = temp1.groupby(['Cycle_Index'], group_keys=True).apply(my_fn)
    if slope:
        temp3 = pd.pivot_table(temp2, values=feat + '_slope', columns='Cycle_Index', index='temp_index').reset_index()
        temp3 = temp3[2:]
        title_ = f'Section {section} slopes'
        if feat == 'Voltage(V)':
            y_label_ = 'Voltage slope (dV/dt)'
        else:
            y_label_ = 'Temperature slope (dT/dt)'

    else:
        temp3 = pd.pivot_table(temp2, values=feat, columns='Cycle_Index', index='temp_index').reset_index()
        title_ = f'Section {section}'
        y_label_ = feat
    time_step = int(re.findall(r'\d+', args.resample_time)[0])

    temp3['Time(h)'] = temp3.temp_index * time_step / 3600

    plot_sections_aux(temp3, args, ser=ser, y_label=y_label_, title=title_, colormap=colormap, offset=offset,
                      offset_value=offset_value, feat=feat)


def plot_sections_aux_3_full(df, args, ser='b1c0', feat='Voltage(V)', colormap='bwr'):
    if feat == 'Voltage(V)':
        offset_value = 2
    else:
        offset_value = 8

    time_step = int(re.findall(r'\d+', args.resample_time)[0])

    temp1 = df[(df.serial_no == ser)]
    cycles_list = temp1.Cycle_Index.unique()
    range_ = range(int(min(cycles_list)), int(max(cycles_list)), int((max(cycles_list) - min(cycles_list)) / 6))
    f_cyc_list = []
    for i in range_:
        f_cyc_list.append(min(cycles_list, key=lambda x: abs(x - i)))
    f_cyc_list = list(set(f_cyc_list))
    f_cyc_list.sort()
    no_plots = len(f_cyc_list) - 1

    color_list = list(range(0, int(f_cyc_list[-1]), int(int(f_cyc_list[-1]) / (no_plots + 1))))[1:]

    NUM_COLORS = f_cyc_list[-1]

    font_text = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 14}
    fonts_main = {'family': 'Arial',
                  'weight': 'regular',
                  'size': 12}
    matplotlib.rc('font', **fonts_main)

    cm = plt.get_cmap(colormap)
    cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(color_list[i]) for i in range(len(f_cyc_list))])

    for i in range(len(f_cyc_list)):
        temp2 = temp1[(temp1.Cycle_Index == f_cyc_list[i])]
        temp2.reset_index(inplace=True, drop=True)
        temp2 = temp2.assign(Time=(temp2.index * time_step / 3600).values)
        temp2.rename(columns={'Time': 'Time(h)'}, inplace=True)

        ax.plot(temp2['Time(h)'], temp2[feat] + i * offset_value, linewidth=3.0)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    plt.xticks(**fonts_main)  # This argument will change the font.
    plt.yticks(**fonts_main)  # This argument will change the font.

    plt.xlabel('Time(h)', **font_text)
    plt.ylabel(f'Offsetted {feat} Values', **font_text)

    plt.title(f'Cell number ({ser})', **font_text)
    cbar = plt.colorbar(scalarMap)
    cbar.ax.set_title('Cycles', **font_text)

    cbar.set_ticks(color_list)
    # cbar.set_ticklabels(tick_label_list)
    cbar.set_ticklabels(f_cyc_list)
    plt.tight_layout()
    plot_path = args.plot_path + 'sections/'
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_path + f'Sections_serial({ser})_{feat}_full.jpg', dpi=600)
    plt.show()

    plt.close()


def plot_sections(final_df, args, conds, plot_ser='b1c0'):
    a = 2
    '''
    Sections:
    I: Constant current (Charging)
    II: Constant voltage (Charging)
    III: Rest after charging
    IV: Discharge
    V: Rest After discharge
    '''
    ser = plot_ser
    col_list = ['Blues']
    feat_list = ['Voltage(V)']
    for i in range(len(col_list)):
        # colormap = 'RdYlGn_r'
        colormap = col_list[i]
        # colormap = 'Blues'
        # colormap = 'BuPu'
        # colormap = 'bwr'
        # feat = 'Voltage(V)'
        feat = feat_list[i]
        # *************************************
        # Group by Cycles -- Total -- Section Full
        plot_sections_aux_3_full(final_df, args, ser=ser, feat=feat, colormap=colormap)

        # *************************************
        # Group by Cycles -- Total -- Section I
        plot_sections_aux_2(final_df, conds, args, section='I', ser=ser, feat=feat, slope=False, colormap=colormap,
                            offset=False,
                            offset_value=0.05)

        # *************************************
        # Group by Cycles -- Total -- Section I -- Slopes
        plot_sections_aux_2(final_df, conds, args, section='I', ser=ser, feat=feat, slope=True, colormap=colormap,
                            offset=True,
                            offset_value=0.05)

        # *************************************
        # Group by Cycles -- Total -- Section II
        plot_sections_aux_2(final_df, conds, args, section='II', ser=ser, feat=feat, slope=False, colormap=colormap,
                            offset=True,
                            offset_value=0.05)

        # *************************************
        # Group by Cycles -- Total -- Section II -- Slopes
        plot_sections_aux_2(final_df, conds, args, section='II', ser=ser, feat=feat, slope=True, colormap=colormap,
                            offset=True,
                            offset_value=0.05)

        # *************************************
        # Group by Cycles -- Total -- Section III
        plot_sections_aux_2(final_df, conds, args, section='III', ser=ser, feat=feat, slope=False, colormap=colormap,
                            offset=False, offset_value=0.05)

        # *************************************
        # Group by Cycles -- Total -- Section III -- Slopes
        plot_sections_aux_2(final_df, conds, args, section='III', ser=ser, feat=feat, slope=True, colormap=colormap,
                            offset=True,
                            offset_value=0.05)

        # *************************************
        # Group by Cycles -- Total -- Section V
        plot_sections_aux_2(final_df, conds, args, section='V', ser=ser, feat=feat, slope=False, colormap=colormap,
                            offset=False,
                            offset_value=0.05)

        # *************************************
        # Group by Cycles -- Total -- Section V -- Slopes
        plot_sections_aux_2(final_df, conds, args, section='V', ser=ser, feat=feat, slope=True, colormap=colormap,
                            offset=True,
                            offset_value=0.05)


def plot_CV(final_df, test_df, serial_no_list_test, args):
    plot_path = args.main_path + f'plots/train_CV/'
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    for ser in serial_no_list_test:
        fig = plt.figure()
        aa = final_df.loc[final_df['serial_no'] == ser]
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        # Add some extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.3)

        ax1.plot(test_df.loc[test_df['serial_no'] == ser, 'datetime_c2_max'],
                 test_df.loc[test_df['serial_no'] == ser, 'Capacity'], 'o-', label='Real')
        ax1.plot(test_df.loc[test_df['serial_no'] == ser, 'datetime_c2_max'],
                 test_df.loc[test_df['serial_no'] == ser, f'pred_Capacity'], '^-', label='Predicted')
        ax1.set_xlabel(r"Date")

        new_tick_locations = np.array([.2, .5, .9])

        # Move twinned axis ticks and label from top to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        # Offset the twin axis below the host
        ax2.spines["bottom"].set_position(("axes", -0.28))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)

        ax2.spines["bottom"].set_visible(True)

        ax2.plot(test_df.loc[test_df['serial_no'] == ser, 'Cycle_Index'],
                 test_df.loc[test_df['serial_no'] == ser, 'Capacity'], alpha=0)  # Create a dummy plot
        ax2.set_xlabel(r"Cycles")

        plt.title(f'CV_train_Capacity_serial_no({ser})')
        for ax in [ax1, ax2]:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        ax1.legend()

        ax1.set_ylabel(f'Capacity (Ah)')
        plt.tight_layout()
        plt.savefig(plot_path + f'final_results_Capacity_multi_feat_serial_no({ser}).png')
        plt.show()
        plt.close()


def plot_final_results(df_input, args):
    plot_path = args.plot_path + 'final_results/'
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    np.sort(df_input.serial_no.unique())
    serial_no_list = df_input.serial_no.unique()

    no_of_plots = 2

    kf = KFold(n_splits=no_of_plots, shuffle=True, random_state=args.random_state)
    kf.get_n_splits(serial_no_list)

    font_text = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 14}
    fonts_main = {'family': 'Arial',
                  'weight': 'regular',
                  'size': 12}
    matplotlib.rc('font', **fonts_main)

    ser_list = [None] * no_of_plots

    i = 0

    for list1_index, list2_index in kf.split(serial_no_list):
        ser_list[0], ser_list[1] = serial_no_list[list1_index], serial_no_list[list2_index]

    for i in range(2):
        color = iter(mplcm.tab20(np.linspace(0, 1, len(ser_list[i]))))
        plt.figure()
        for ser in ser_list[i]:
            temp = df_input[df_input.serial_no == ser]
            temp.reset_index(drop=True, inplace=True)

            clr = next(color)
            line1, = plt.plot(temp.Cycle_Index, temp.Capacity, linewidth=3.0, c=clr, label=ser)
            line2, = plt.plot(temp.Cycle_Index, temp.pred_Capacity, '--', linewidth=3.0, c=clr)

        # Create a legend for the first line.
        first_legend = plt.legend([line1, line2], ['Real', 'Predicted'], loc='lower left')

        # Add the legend manually to the current Axes.
        ax = plt.gca().add_artist(first_legend)
        ax = plt.gca()
        # ax.set_ylim([0.85, 1.1])

        plt.xlabel('Cycles', **font_text)
        plt.ylabel('Capacity (Ah)', **font_text)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(plot_path + f'final_results_list({i}).jpg', dpi=600)
        plt.show()

        plt.close()


def plot_features(df_input, args, plot_ser='b1c0'):
    plot_dir = args.plot_path + f'/features/ser_{plot_ser}/'
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    temp = df_input[df_input.serial_no == plot_ser]
    features_main_list = [x for x in list(temp.columns) if
                          not x in ['serial_no', 'datetime_c2_max', 'Capacity', 'Cycle_Index',
                                    'initial_capacity', 'last_value']]
    df = pd.DataFrame()
    df['features_main'] = pd.Series(features_main_list)

    df = pd.DataFrame(df.features_main.str.split('_').tolist(), columns=['features_variable', 'features_section',
                                                                         'features_slope_type',
                                                                         'features_stat_type'])
    df.loc[(df.features_slope_type != 'slope') & (df.features_slope_type != 'time'), 'features_stat_type'] = df.loc[
        (df.features_slope_type != 'slope') & (df.features_slope_type != 'time'), 'features_slope_type']
    df.loc[(df.features_slope_type != 'slope'), 'features_slope_type'] = 'not_slope'
    df['features_main'] = pd.Series(features_main_list)
    df = df.sort_values(["features_variable", "features_section", "features_slope_type", "features_stat_type"])

    for var in df.features_variable.unique():
        for sec in df.features_section.unique():
            for slp in df.features_slope_type.unique():
                temp_df_2 = df[(df.features_variable == var) & (df.features_section == sec) &
                               (df.features_slope_type == slp)]
                temp_df_2.reset_index(drop=True, inplace=True)
                features_main_list2 = features_main_list.copy()
                features_main_list2.append('Capacity')
                normalized_df = (temp[features_main_list2] - temp[features_main_list2].min()) / \
                                (temp[features_main_list2].max() - temp[features_main_list2].min())
                normalized_df = normalized_df.assign(Cycles=temp.Cycle_Index.values)
                normalized_df.reset_index(drop=True, inplace=True)

                font_text = {'family': 'Times New Roman',
                             'weight': 'bold',
                             'size': 14}
                fonts_main = {'family': 'Arial',
                              'weight': 'regular',
                              'size': 12}
                matplotlib.rc('font', **fonts_main)

                corr_df = temp.corr()
                corr_df = corr_df.round(decimals=2)
                corr_df = corr_df.fillna(0)
                temp_arr = mutual_info_regression(temp[features_main_list], temp['Capacity'],
                                                  random_state=args.random_state)
                mutinf_df = pd.DataFrame(temp_arr).transpose()
                mutinf_df.set_axis(features_main_list, axis=1, inplace=True)
                markers = ['--o', '--v', '--*', '--h', '--x', '--D', '--s', '--p', '--h']

                if var == 'Voltage(V)':
                    color = iter(mplcm.Blues(np.linspace(0, 1, len(temp_df_2) + 3)))
                    clr = next(color)
                    clr = next(color)
                    clr = next(color)
                else:
                    color = iter(mplcm.Greens(np.linspace(0, 1, len(temp_df_2) + 3)))
                    clr = next(color)
                    clr = next(color)
                    clr = next(color)

                plt.figure()
                for plt_num in range(len(temp_df_2)):
                    clr = next(color)
                    plt.plot(normalized_df.Cycles, normalized_df[temp_df_2.loc[plt_num, 'features_main']],
                             markers[plt_num],
                             linewidth=3.0, c=clr,
                             label=f"{temp_df_2.loc[plt_num, 'features_stat_type']}({mutinf_df[temp_df_2.loc[plt_num, 'features_main']][0]:.2f})")
                plt.plot(normalized_df.Cycles, normalized_df.Capacity, '--', linewidth=3.0, c='red', label='Capacity')
                plt.xlabel('Cycles', **font_text)
                plt.ylabel('Normalized values', **font_text)
                if slp == 'slope':
                    temp_str = slp + 's'
                else:
                    temp_str = ''

                plt.title(f'Cell number ({plot_ser}) {var} section {sec} {temp_str}', **font_text)
                plt.legend(loc=3)
                plt.tight_layout()
                plt.savefig(plot_dir + f'features_serial({plot_ser})_{var}_{temp_str}_{sec}.jpg', dpi=600)
                plt.show()

                plt.close()


def feat_imp_plot_region(df_in, args):
    df_in = df_in.sort_values(by='sum_imp', ascending=False)
    plot_path = args.plot_path + 'feat_importance/'
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    font_text = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 14}
    fonts_main = {'family': 'Arial',
                  'weight': 'regular',
                  'size': 12}
    matplotlib.rc('font', **fonts_main)

    col = df_in.columns[0]
    plt.figure()
    ax = df_in.plot.bar(x=col, y='sum_imp', rot=0, legend=False)
    # ax = df_in[['sum_imp']].plot(kind='bar')
    ax.set_ylabel("Sum of Importance")
    plt.tight_layout()
    plt.savefig(plot_path + f'features_importance_region_{col}.png', dpi=600)
    plt.show()
    plt.close()


def feat_imp_plot(df_in, args, top_number=10):
    plot_path = args.plot_path + 'feat_importance/'
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    df = df_in[:top_number]
    # df.shap_importance = df.shap_importance / df.shap_importance.max()
    df = df.assign(shap_importance=df.shap_importance / df.shap_importance.max())
    df = df.sort_values(by='shap_importance')

    font_text = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 14}
    fonts_main = {'family': 'Arial',
                  'weight': 'regular',
                  'size': 12}
    matplotlib.rc('font', **fonts_main)

    fig, ax1 = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.115, right=0.88)
    fig.canvas.set_window_title('Eldorado K-8 Fitness Chart')

    pos = np.arange(len(df))

    rects = ax1.barh(pos - len(pos), [df.loc[k, 'shap_importance'] for k in df.index],
                     align='center',
                     height=0.5,
                     tick_label=df['column_name'])
    ax1.set_xlabel('Normalized values')
    plt.title('Features importance')
    plt.tight_layout()
    plt.savefig(plot_path + f'features_importance.png', dpi=600)
    plt.show()
    plt.close()
