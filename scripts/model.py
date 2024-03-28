# Import Libraries
import pandas as pd
from xgboost import Booster, DMatrix
import re

booster = Booster()

# FP
def prediction_fp(date_input):
    try:
        booster.load_model("file:///Users/bingfeng/Documents/Projects/air_selangor/SIMS/prediction-api/models/fp_model.bin")
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
        booster.load_model("file:///Users/bingfeng/Documents/Projects/air_selangor/SIMS/prediction-api/models/mc_model.bin")
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
        booster.load_model("file:///Users/bingfeng/Documents/Projects/air_selangor/SIMS/prediction-api/models/ml_model.bin")
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
