import os
import math
import joblib
import pandas as pd
import datetime as dt
import numpy as np
import dataiku

def predict_at_proba_threshold(model, X, threshold):
    """
    Predict probability and target 1 or 0 value using probabilities and threshold values;
    We only predict 1, if the threshold is surpassed

    Parameters
    ----------
    model : Single trained model
    X : DataFrame consisting of all features
    threshold : dict with all thresholds
    Returns
    -------
    Y_pred : Series containing 1 or 0 predictions for model for all customers
    Y_prob : Series containing probability predictions for model for all customers
    """

    Y_proba = model.predict_proba(X)
    Y_pred = np.array([1 if proba[1] > threshold else 0 for proba in Y_proba])
    Y_prob = [i[1] for i in Y_proba]
    return Y_pred, Y_prob


def load_most_recent_model(path_to_models):
    """
    Load most recent model from path provided

    Parameters
    ----------
    path_to_models : path where models are stored in DSS
                    (e.g. '/home/dataiku/dss/managed_datasets/v3gGTR/8HIbPjXK/')
    Returns
    -------
    rfc_models : Dict with trained models
    """

    # Get names of all models
    all_models = sorted(os.listdir(path_to_models), reverse=True)

    # Get name of most recent model
    most_recent_model = all_models[0]

    # Load model
    rfc_models = joblib.load(os.path.join(path_to_models, most_recent_model))

    return rfc_models

def predict_probabilities(X_test, rfc_models, dict_thresholds, class_labels):
    """
    Predict probabilities and binary 0 or 1 for all customers in X_test set using all required models

    Parameters
    ----------
    X_test : DataFrame with all test features that we will predict for
    rfc_models : dict with all trained models
    dict_thresholds :  thresholds which are used to convert probabilities into 0 or 1 for each model
    class_labels : list with class labels (e.g. 'is_2M')

    Returns
    -------
    df_preds : DataFrame with predictions
    """
    df_preds = pd.DataFrame()

    for target in class_labels:
        df_preds["pred_"+ target], df_preds["proba_"+ target] = predict_at_proba_threshold(rfc_models[target], X_test, dict_thresholds[target])

    return df_preds

def get_final_prediction_from_probabilities(df_preds, default_prediction_value):
    """
    Determine final prediction from individual models predictions (models being 2M, 3M, etc)

    Parameters
    ----------
    df_preds : DataFrame with probability predictions
    default_prediction_value : If no model predicts 1, then choose a default value

    Returns
    -------
    df_preds : DataFrame probability predictions as well as final prediction
    """
    df_preds["predicted"] = df_preds.apply(lambda row: get_predicted(row, default_prediction_value), axis = 1)
    return df_preds

def get_predicted(row, default_prediction_value):
    """
    Combine predictions from each model into one final prediction;
    This is done by choosing the lowest model (e.g. M2) where predicted is equal to 1
    (the lowest to be conservative)

    Parameters
    ----------
    row : One row of df_preds
    default_prediction_value : If no model predicts 1, then choose a default value

    Returns
    -------
    pred : One single predicted value (e.g. 4)
    """

    pred = default_prediction_value
    if row["pred_is_2M"]==1:
        pred = 2
    elif row["pred_is_3M"]==1:
        pred = 3
    elif row["pred_is_4M"]==1:
        pred = 4
    elif row["pred_is_5M"]==1:
        pred = 5
    elif row["pred_is_6M"]==1:
        pred = 6
    elif row["pred_is_7M"]==1:
        pred = 7
    elif row["pred_is_8M"]==1:
        pred = 8
    elif row["pred_is_9M"]==1:
        pred = 9
    elif row["pred_is_10M"]==1:
        pred = 10
    elif row["pred_is_11M"]==1:
        pred = 11
    elif row["pred_is_12M_plus"]==1:
        pred = 12
    return pred

def join_cust_info_to_preds(df_preds, df):
    """
    Join customer information (customer id, access start date, territory)
    onto prediction DataFrame

    Parameters
    ----------
    df_preds : DataFrame with predictions

    Returns
    -------
    df_ect_12m : DataFrame with both predictions and joined customer information
    """

    identifier_columns = ['cust_account_id', 'access_start_date', 'cust_territory', 'cust_country']
    prediction_columns = list(df_preds.columns.values)

    df_ect_12m = df_preds.join(df[identifier_columns])[identifier_columns + prediction_columns]

    return df_ect_12m

def add_pred_date_and_inserted_at_to_pred_df(df_ect_12m):
    """
    Add date of prediction and date inserted into the table into the prediction DataFrame

    Parameters
    ----------
    df_preds : DataFrame with predictions

    Returns
    -------
    df_ect_12m : DataFrame with both predictions and joined customer information
    """

    prediction_date = dataiku.get_custom_variables()['calculation_date']
    df_ect_12m['prediction_date'] = prediction_date
    df_ect_12m['inserted_at'] = dt.datetime.today().strftime('%Y-%m-%d %H:%M')

def save_preds_to_table(df_ect_12m, table_name):
    """
    Save predictions to an s3 bucket

    Parameters
    ----------
    df_preds : DataFrame with predictions

    Returns
    -------
    df_ect_12m : DataFrame with both predictions and joined customer information
    """

    train_model_diagnostics = dataiku.Dataset(table_name)
    train_model_diagnostics.write_with_schema(df_ect_12m)
