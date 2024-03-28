# Import Libraries

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
        booster.load_model(fp_model_path)  # Ensure booster is defined and fp_model_path is set
        pattern = re.compile(r'^\d{4}-\d{2}$')
        
        if not pattern.match(date_input):
            return 'The date input format is invalid.'
        
        # Parse the date and extract year and month
        year, month = map(int, date_input.split('-'))
        
        # Create a list of tuples
        data = [(month, year)]
        
        # Create DMatrix
        predict_dmatrix = DMatrix(data)
        
        # Perform prediction
        result = booster.predict(predict_dmatrix)
        return result
    except Exception as e:
        return str(e)

# MC
def prediction_mc(date_input):
    try:
        booster.load_model(mc_model_path)
        pattern = re.compile(r'^\d{4}-\d{2}$')
        
        if not pattern.match(date_input):
            return 'The date input format is invalid.'
        
        # Parse the date and extract year and month
        year, month = map(int, date_input.split('-'))
        
        # Create a list of tuples
        data = [(month, year)]
        
        # Create DMatrix
        predict_dmatrix = DMatrix(data)
        
        # Perform prediction
        result = booster.predict(predict_dmatrix)
        return result
    except Exception as e:
        return str(e)
# ML
def prediction_ml(date_input):
    try:
        booster.load_model(ml_model_path)
        pattern = re.compile(r'^\d{4}-\d{2}$')

        if not pattern.match(date_input):
            return 'The date input format is invalid.'
        
        # Parse the date and extract year and month
        year, month = map(int, date_input.split('-'))
        
        # Create a list of tuples
        data = [(month, year)]
        
        # Create DMatrix
        predict_dmatrix = DMatrix(data)
        
        # Perform prediction
        result = booster.predict(predict_dmatrix)
        return result
    except Exception as e:
        return str(e)

