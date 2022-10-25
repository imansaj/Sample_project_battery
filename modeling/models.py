import numpy as np
import pandas as pd
import shap as shap
from lightgbm import LGBMRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

from plots.plots import plot_CV, feat_imp_plot_region, feat_imp_plot


class MainModel:
    """
    The main model class which create the train, test data using Kfold method, picks the top features using mutual
    information and applies the model to the data, and save the results.

    :param final_file: The final pandas is saved to a pickle for further use.
    :param args: The global variables tha are passed around in the prohect.
    :param no_features: The number of features used for the final model.
    """
    def __init__(self, args, no_features=15):

        self.args = args
        self.random_state = args.random_state
        self.no_features = no_features

    def train_cv(self, df, splits=5, plot_flag=True, plot_importance=True):
        """
        The function that applies cross validation algorithm to the data.
        :param df: The input data
        :param splits: Number of splits used for kfold.
        :param plot_flag: Eohter we wat to plto the data or not.
        :param plot_importance: Either we wnat to plto the feature importance or not.
        :return: Returns a pandas which has the results of the model applied to all cv parts.
        """
        serial_no_list = df.serial_no.unique()

        input_columns = [x for x in list(df.columns) if
                         not x in ['serial_no', 'datetime_c2_max', 'Capacity', 'Cycle_Index']]

        kf = KFold(n_splits=splits, shuffle=True, random_state=self.random_state)
        kf.get_n_splits(serial_no_list)

        loss_columns = []
        score_columns = []
        final_df_list = []

        k = 0
        feat_df = pd.DataFrame()
        for train_index, test_index in kf.split(serial_no_list):
            loss_columns.append(f'loss{k}')
            score_columns.append(f'score{k}')
            serial_no_list_train, serial_no_list_test = serial_no_list[train_index], serial_no_list[test_index]
            feat_num = self.no_features

            train_df = df[df.serial_no.isin(serial_no_list_train)]
            test_df = df[df.serial_no.isin(serial_no_list_test)]

            train_df = train_df.fillna(0)
            test_df = test_df.fillna(0)

            test_df = test_df.sort_values(['serial_no', 'datetime_c2_max'])

            # **********************************************
            # Feature Selection

            mask = self.feature_selector(train_df, input_columns, feat_num=feat_num)
            X_train = train_df[mask]
            y_train = train_df['Capacity']
            self.model = self.model_()
            self.model.fit(X_train.values, y_train)

            test_df, score, loss = self.predict(test_df, mask)
            feat_df.loc[feat_num, f'score{k}'] = score
            feat_df.loc[feat_num, f'loss{k}'] = loss
            final_df_list.append(test_df)
            if plot_flag:
                plot_CV(df, test_df, serial_no_list_test, self.args)
            k = k + 1

        if plot_importance:
            self.plot_importance(train_df, mask)
        final_cv_df = pd.concat(final_df_list)
        feat_df = feat_df.fillna(0)
        feat_df['mean_loss'] = feat_df[loss_columns].mean(axis=1)
        feat_df['std_loss'] = feat_df[loss_columns].std(axis=1)
        feat_df.reset_index(inplace=True, drop=True)

        print(f'The method is cross validated on {splits} parts.')
        print("The mean of RMSE is %0.5f with a standard deviation of %0.5f" % (feat_df.mean_loss.max(),
                                                                                feat_df.loc[
                                                                                    feat_df.mean_loss.idxmax(), 'std_loss']))
        return final_cv_df

    def model_(self):
        """
        Returns the desired model.
        :return:
        """
        mod = LGBMRegressor()
        return mod

    def plot_importance(self, df, mask):
        """
        Plots the importance of each feature. The features used for this function are the ones selected by mutual
        information.
        :param df: The train dataset
        :param mask: Top features
        """
        X_train_df = df[mask]
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_train_df)

        shap_sum = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame([mask, shap_sum.tolist()]).T
        importance_df.columns = ['column_name', 'shap_importance']
        importance_df = importance_df.sort_values('shap_importance', ascending=False)
        importance_df.reset_index(inplace=True, drop=True)
        check = importance_df.copy()
        check = check.iloc[1:]
        check.reset_index(inplace=True, drop=True)
        check[['Feature', 'temp']] = check['column_name'].str.split('_', 1, expand=True)
        check[['Section', 'temp']] = check['temp'].str.split('_', 1, expand=True)
        check.loc[~check['temp'].str.contains("_"), 'temp'] = 'notSlope_' + check.loc[
            ~check['temp'].str.contains("_"), 'temp']
        check[['Slope', 'Stat']] = check['temp'].str.split('_', 1, expand=True)
        section_imp = check.groupby(['Section']).agg(sum_imp=('shap_importance', 'sum')).reset_index()
        slope_imp = check.groupby(['Slope']).agg(sum_imp=('shap_importance', 'sum')).reset_index()
        stat_imp = check.groupby(['Stat']).agg(sum_imp=('shap_importance', 'sum')).reset_index()
        feat_imp_plot_region(section_imp, self.args)
        feat_imp_plot_region(slope_imp, self.args)
        feat_imp_plot_region(stat_imp, self.args)
        feat_imp_plot(importance_df, self.args)

    def predict(self, X, mask):
        """

        :param X:
        :param mask:
        :return:
        """
        X = X.sort_values(['serial_no', 'datetime_c2_max'])
        y = X['Capacity']

        # *******************************************************************
        # Predicting capacity using last known capacity
        # *******************************************************************
        def apply_last_value(df, cols, output):
            for i in df.index:
                try:
                    df.loc[i, f'pred_{output}'] = \
                        self.model.predict(df[cols].iloc[i - df.index[0]].values.reshape(1, -1))[
                            0]
                except:
                    a = 3
                if i < df.index[-1]:
                    df.loc[i + 1, 'last_value'] = df.loc[i, f'pred_{output}']
            return df

        # X = X.assign(pred_capacity=0)
        X[f'pred_Capacity'] = 0
        X = X.reset_index(drop=True)

        X = X.groupby(['serial_no']).apply(apply_last_value, cols=mask, output='Capacity')
        X = X.reset_index(drop=True)

        return X, r2_score(X[f'pred_Capacity'], y), mean_squared_error(X[f'pred_Capacity'], y, squared=False)

    def feature_selector(self, train_df, input_columns, feat_num=20):
        # Feature Selection
        def df_mut_inf(df, col_list, args, random_state):
            temp_arr = mutual_info_regression(df[col_list], df['Capacity'], random_state=random_state)
            temp_df = pd.DataFrame(temp_arr).transpose()
            temp_df.set_axis(input_columns, axis=1, inplace=True)
            temp_df['serial_no'] = df['serial_no'].unique()[0]
            return temp_df

        mut_inf_df = train_df.groupby('serial_no').apply(df_mut_inf, input_columns, self.args, self.random_state)
        mut_inf_df.drop(['serial_no'], axis=1, inplace=True)

        mean_df = mut_inf_df.mean(axis=0, numeric_only=None)

        return list(mean_df.sort_values(ascending=False).head(feat_num).index)
