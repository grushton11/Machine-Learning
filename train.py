import matplotlib
matplotlib.use("Agg")

import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import numpy as np

import datetime as dt
from dateutil import parser
from dateutil.relativedelta import relativedelta
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

import eli5
from eli5.sklearn import PermutationImportance
from eli5.sklearn import explain_prediction_linear_classifier

import math
import os
import joblib

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    return percentile_

def convert_data_types(df):
    df['access_start_date'] = pd.to_datetime(df['access_start_date'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def create_new_features(df):
    df['access_start_month'] = df['access_start_date'].dt.month

def _payment_type_mapping(var_value):
    """
    Map single payment type (e.g. Bank Transfer and Bank Transfer both to Bank Transfer)

    Parameters
    ----------
    var_value : Single payment type value (e.g. CreditCard)
    Returns
    -------
    var_value : Correctly mapped payment type value (e.g. Credit Card)
    """

    if var_value in ['CreditCard', 'CreditCardReferenceTransaction']:
        return 'Credit Card'
    if var_value in ['BankTransfer', 'Bank Transfer']:
        return 'Bank Transfer'
    if var_value in ['Amazon', 'Amazon Pay']:
        return 'Amazon'
    if var_value in ['Apple', 'Apple Pay']:
        return 'Apple'
    else:
        return var_value

def prepare_feature_payment_type(df, payment_method_variable_name):
    """
    Map payment types (e.g. Bank Transfer and Bank Transfer both to Bank Transfer)

    Parameters
    ----------
    df : DataFrame consisting all training data
    Returns
    -------
    payment_method_feature_list : List of payment features to use in modelling
    """

    payment_method_unfiltered = ['Apple', 'Apple Pay', 'Credit Card', 'CreditCard', 'CreditCardReferenceTransaction', 'BankTransfer', 'Bank Transfer', 'PayPal', 'Amazon', 'Amazon Pay', 'Direct Debit']
    payment_method_feature_list = ['Apple', 'Credit Card', 'Bank Transfer', 'PayPal', 'Amazon', 'Direct Debit']

    df[payment_method_variable_name] = df[payment_method_variable_name].apply(lambda x: _payment_type_mapping(x) if x in payment_method_unfiltered else np.nan)
    return payment_method_feature_list

def prepare_feature_preferred_sport(df, preferred_sport_list):
    """
    Only use sports which are predefined in preferred_sport_list

    Parameters
    ----------
    df : DataFrame consisting of all data
    preferred_sport_list: List with all preferred sports which should be used for modelling

    Returns
    -------
    payment_method_feature_list : List of payment features to use in modelling
    """

    df['preferred_sport_by_hours'] = df['preferred_sport_by_hours'].apply(lambda x: x if x in preferred_sport_list else np.nan)
    return preferred_sport_list


def imputate_nans(df, vars_to_impute, imputations_dict):
    """
    Automatically impute variables (vars_to_impute) by specified values (imputations_dict)

    Parameters
    ----------
    df : DataFrame consisting of all data
    vars_to_impute: List with variables that we want to impute
    imputations_dict: dictionary specifyfing how to impute variables in vars_to_impute list

    Returns
    -------
    None; Imputation is applied inplace
    """

    imputations_list = list(imputations_dict.keys())
    if not set(vars_to_impute).issubset(imputations_list):
        raise ValueError('Please specify an imputation in imputations_dict for all variables in vars_to_impute.')


    for var in vars_to_impute:
        if var == 'months_to_competition_end':
            df["months_to_competition_end"] = df["months_to_competition_end"].apply(lambda x: imputations_dict['months_to_competition_end'] if (math.isnan(x)) | (x > 12) else x)

        else:
            df[var] = df[var].fillna(imputations_dict[var])

def create_binary_targets(df, start_month):
    """
    Create binary target variables (e.g. 'is_2M') in df and return all targets in list

    Parameters
    ----------
    df : DataFrame consisting of all data
    start_month: start month for which target variables should be created;
                 e.g. 6 would create targets from month 6 to 12

    Returns
    -------
    all_class_labels : list with created class labels
    """

    # # create new binary targets, one for each possible class
    start_month = start_month - 1

    class_labels = []

    m12_label = 'is_12M_plus'
    df[m12_label] = df['tenure_length_capped'].apply(lambda x: 1 if x == 12 else 0)

    for target_month in range(11, start_month, -1):
        temp_month_label = 'is_{}M'.format(target_month)
        df[temp_month_label] = df['tenure_length_capped'].apply(lambda x: 1 if x == target_month else 0)

        class_labels.append(temp_month_label)

    all_class_labels = [m12_label] + class_labels
    return df, all_class_labels

def create_dummy_variables(df, vars_list_to_create_dummies):
    """
    Dummify variables and add them to dataframe

    Parameters
    ----------
    df : DataFrame consisting of all data
    vars_list_to_create_dummies : list of all variables that should be dummified

    Returns
    -------
    all_class_labels : list with created class labels
    """

    for var in vars_list_to_create_dummies:
        print('Create dummy for {}'.format(var))

        df_temp_dummy =  pd.get_dummies(df[var])
        df_temp_dummy.reset_index()
        df = df.drop(var, axis=1)
        df = pd.concat([df, df_temp_dummy], axis=1)

    return df

def get_train_data(df_features, synchronization_time_days, model_diagnostics, class_labels):
    """
    Get train data for each model (most recent available 12 months) from all training data

    Parameters
    ----------
    df_features : DataFrame consisting of all data required for training (dummies, numeric features)
    synchronization_time_days : We know that data is not directly coming through;
                                being very conservative, it is set at 30 days
    model_diagnostics : dictionary into which diagnostics in training process will be inserted

    Returns
    -------
    all_class_labels : list with created class labels
    """

    train_date = dataiku.get_custom_variables()['calculation_date']

    df_features_train = {}
    for k, i in enumerate(class_labels):
        target_month = 12 - k


        # create end and start period for training
        training_end_date = parser.parse(train_date) + relativedelta(months=-target_month, days = - synchronization_time_days)
        training_start_date = parser.parse(train_date) + relativedelta(years=-1, months=-target_month, days = - synchronization_time_days)

        # convert datetime to string
        training_start_date_str = training_start_date.strftime('%Y-%m-%d')
        training_end_date_str = training_end_date.strftime('%Y-%m-%d')

        start_mask = df_features['access_start_date'] >= training_start_date_str
        end_mask = df_features['access_start_date'] <= training_end_date_str
        status_unknown = (df_features['tenure_length_capped'] == target_month) & (df_features['is_churn'] == 0)

        df_features_train_sampled = df_features.loc[start_mask & end_mask & ~status_unknown]

        df_features_train[i] = df_features_train_sampled

        model_diagnostics[i]['training_samples_available'] = df_features_train_sampled.shape[0]
        model_diagnostics[i]['train_start_date'] = df_features_train_sampled['access_start_date'].min().strftime('%Y-%m-%d')
        model_diagnostics[i]['train_end_date'] = df_features_train_sampled['access_start_date'].max().strftime('%Y-%m-%d')

        print(i + ': train_samples: ' + str(df_features_train_sampled.shape[0]) + '; start_time:', df_features_train_sampled['access_start_date'].min().strftime('%Y-%m-%d'), '; end_time:', df_features_train_sampled['access_start_date'].max().strftime('%Y-%m-%d'))

    return df_features_train, model_diagnostics

def train_rfc_cascade(class_labels_to_train, df_features_train, model_diagnostics):
    """
    Downsample data, train model and output diagnostics into model_diagnostics dict

    Parameters
    ----------
    class_labels_to_train : list with class labels (e.g. 'is_2M')
    df_features_train : DataFrame with all train features and additional columns
                        which will be dropped (e.g. access_start_date, is_churn)
    model_diagnostics : dictionary into which diagnostics in training process will be inserted

    Returns
    -------
    rfc_models : dict where keys are class labels with trained models
    """

    rfc_models = []
    for k, target in enumerate(class_labels_to_train):
        print("started work on target: " + target)

        # class labels to drop
        class_labels_to_drop = list(class_labels_to_train)
        class_labels_to_drop.remove(target)

        df_features_train_prepared = df_features_train[target]

        # Downsample
        class_weight = df_features_train_prepared[target].sum() / 1.0 / df_features_train_prepared.shape[0]
        print("class weight: " + str(class_weight))
        sample_size = df_features_train_prepared[target].sum()
        df_features_train_sampled = df_features_train_prepared.groupby(target, group_keys=False).apply(lambda group: group.sample(sample_size))

        # Output to console
        total_train = df_features_train_sampled.shape[0]
        total_target_class = df_features_train_sampled[target].sum()
        percentage_target_train = total_target_class / float(total_train)

        print("train sample size     : " + str(total_train))
        print("% target class        : " + str(percentage_target_train))

        # create train X and Y
        X_train = df_features_train_sampled.copy()
        X_train = X_train.drop(class_labels_to_drop, axis=1)
        X_train = X_train.drop(target, axis=1)
        X_train = X_train.drop("access_start_date", axis=1)
        X_train = X_train.drop("tenure_length_capped", axis=1)
        X_train = X_train.drop("is_churn", axis=1)
        Y_train = df_features_train_sampled.loc[:,[
            target
        ]].values.ravel()

        # train model
        rfc = RandomForestClassifier(oob_score=True, random_state = 0, n_estimators=100, min_samples_split=0.01, max_depth=10, min_samples_leaf=0.01)
        print("training started at   : " + str(dt.datetime.now()))
        rfc_model = rfc.fit(X_train, Y_train)
        print("training done with oob: " + str(rfc_model.oob_score_))
        print("")
        rfc_models.append(rfc_model)

        model_diagnostics[target]['oob_score'] = rfc_model.oob_score_
        model_diagnostics[target]['model_parameter'] = rfc

        model_diagnostics[target]['total_train_samples_used'] = total_train
        model_diagnostics[target]['total_train_samples_target_class'] = total_target_class
        model_diagnostics[target]['total_train_percentage_target_class'] = class_weight

        train_preds = rfc_model.predict(X_train)
        model_diagnostics[target]['train_set_accuracy_score'] = accuracy_score(Y_train, train_preds)
        model_diagnostics[target]['train_set_precision_score'] = precision_score(Y_train, train_preds)
        model_diagnostics[target]['train_set_recall_score'] = recall_score(Y_train, train_preds)
        model_diagnostics[target]['train_set_f1_score'] = f1_score(Y_train, train_preds)

    rfc_models = dict(zip(class_labels_to_train, rfc_models))
    return rfc_models

def save_model(rfc_models, dss_folder_id, model_name):
    """
    Save model in DSS folder

    Parameters
    ----------
    rfc_models :  dictionary with all models
    dss_folder_id : id of folder to save model into
    model_name : Name of model to be saved as

    Returns
    -------
    rfc_models : dict where keys are class labels with trained models
    """

    models = dataiku.Folder(dss_folder_id)
    path_to_folder = models.get_path()
    current_time = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    path_to_save_model = os.path.join(path_to_folder, current_time + '_' + model_name)

    path_to_save_compressed_model = path_to_save_model + '_compressed_v01'
    model_save_path = joblib.dump(rfc_models, path_to_save_compressed_model, compress = True)
    print(model_save_path)


def get_feature_importance_from_eli5(df_features_train, features_in_model, rfc_models, target_month, sample_size):
    """
    Calculate Permutated Feature Importance for one model

    Parameters
    ----------
    df_features_train : train feature set in dictionary form
    features_in_model : All features which are used to train model
    target_month : Month to perform features importance for (e.g. 'is_2M')
    sample_size : Number of samples to calculate Permutated Feature Importance for

    Returns
    -------
    imp_df : DataFrame which consists of features name and importances (ranging -1 to 1)
    """

    fi_test_data = df_features_train[target_month].sample(sample_size)

    fi_X_test = fi_test_data[features_in_model]
    fi_y_test = fi_test_data[target_month]

    perm = PermutationImportance(rfc_models[target_month], cv = 'prefit', n_iter = 10, scoring='accuracy').fit(fi_X_test, fi_y_test)

    names = fi_X_test.columns.values
    importances = perm.feature_importances_
    imp_df = pd.DataFrame({'importance': importances, 'names': names})
    imp_df.sort_values(by='importance', ascending=True, inplace=True)
    return imp_df

def calculate_feature_importance(df_features_train, features_in_model, rfc_models, class_labels, model_diagnostics, sample_size):
    """
    Calculate Permutated Feature Importance for all models in a for loop
    and ouput feature into model diagnostics dict

    Parameters
    ----------
    class_labels : Class labels for model
    model_diagnostics : dictionary into which diagnostics in training process will be inserted

    Returns
    -------
    model_diagnostics : dictionary into which diagnostics have been inserted
    """

    for target_month in class_labels:

        if df_features_train[target_month].shape[0] < sample_size:
            sample_size = df_features_train[target_month].shape[0]

        imp_df = get_feature_importance_from_eli5(df_features_train, features_in_model, rfc_models, target_month, sample_size)

        # Express as percentage
        imp_df['importance_perc'] = imp_df['importance'] * 100

        # Create dictionary with importances
        feature_importance_dict = dict(zip(imp_df['names'], list(imp_df['importance_perc'])))
        feature_importance_dict['sample_size_permuated_feature_importance'] = sample_size

        model_diagnostics[target_month]['feature_importance'] = dict(zip(imp_df['names'], list(imp_df['importance_perc'])))

        print("Completed feature importance training for target month: " + target_month)
    return model_diagnostics

def save_model_diagnostics(table_name, model_diagnostics):
    """
    Save all model diagnostics in a dataframe

    Parameters
    ----------
    class_labels : Class labels for model
    model_diagnostics : dictionary into which diagnostics in training process will be inserted

    Returns
    -------
    model_diagnostics : dictionary into which diagnostics have been inserted
    """

    diagnostics_df = pd.DataFrame(model_diagnostics).transpose()
    diagnostics_df['model_training_date'] = datetime.today().strftime('%Y-%m-%d %H:%M')

    diagnostics_df['model'] = diagnostics_df.index

    cols = ['model', 'model_training_date', 'train_start_date', 'train_end_date', 'training_samples_available', 'total_train_percentage_target_class', 'total_train_samples_target_class', 'total_train_samples_used', 'oob_score', 'model_parameter', 'feature_importance']
    diagnostics_df = diagnostics_df[cols]

    train_model_diagnostics = dataiku.Dataset(table_name)
    train_model_diagnostics.write_with_schema(diagnostics_df)
