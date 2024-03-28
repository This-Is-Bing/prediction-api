# Import Libraries
import pandas as pd
from xgboost import Booster, DMatrix
import re

import json
with open('config.json') as config_file:
    config = json.load(config_file)

fp_model_path = config['fp_model_path']
mc_model_path = config['mc_model_path']
ml_model_path = config['ml_model_path']

booster = Booster()

# FP
def prediction_fp(date_input):
    try:
        booster.load_model(fp_model_path)
        pattern = re.compile(r'^\d{4}-\d{2}$')
        if not pattern.match(date_input):
            return('The date input format is invalid.')
        else:
            predict_df = pd.DataFrame([date_input], columns=['Date'])
            predict_df = predict_df.set_index('Date')
            predict_df.index = pd.to_datetime(predict_df.index)
            predict_df['month'] = predict_df.index.month
            predict_df['year'] = predict_df.index.year
            predict_dmatrix = DMatrix(predict_df)
            result = booster.predict(predict_dmatrix)
        return (result)
    except Exception as e:
        return (print(e))

# MC
def prediction_mc(date_input):
    try:
        booster.load_model(mc_model_path)
        pattern = re.compile(r'^\d{4}-\d{2}$')
        if not pattern.match(date_input):
            return('The date input format is invalid.')
        else:
            predict_df = pd.DataFrame([date_input], columns=['Date'])
            predict_df = predict_df.set_index('Date')
            predict_df.index = pd.to_datetime(predict_df.index)
            predict_df['month'] = predict_df.index.month
            predict_df['year'] = predict_df.index.year
            predict_dmatrix = DMatrix(predict_df)
            result = booster.predict(predict_dmatrix)
        return (result)
    except Exception as e:
        return (print(e))

# ML
def prediction_ml(date_input):
    try:
        booster.load_model(ml_model_path)
        pattern = re.compile(r'^\d{4}-\d{2}$')
        if not pattern.match(date_input):
            return('The date input format is invalid.')
        else:
            predict_df = pd.DataFrame([date_input], columns=['Date'])
            predict_df = predict_df.set_index('Date')
            predict_df.index = pd.to_datetime(predict_df.index)
            predict_df['month'] = predict_df.index.month
            predict_df['year'] = predict_df.index.year
            predict_dmatrix = DMatrix(predict_df)
            result = booster.predict(predict_dmatrix)
        return (result)
    except Exception as e:
        return (print(e))
