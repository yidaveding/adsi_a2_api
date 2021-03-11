from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

'''
‘/’ (GET): Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project
‘/health/’ (GET): Returning status code 200 with a string with a welcome message of your choice
‘/beer/type/’ (POST): Returning prediction for a single input only
‘/beers/type/’ (POST): Returning predictions for a multiple inputs
‘/model/architecture/’ (GET): Displaying the architecture of your Neural Networks (listing of all layers with their types)
'''

app = FastAPI()

rf_pipe = load('../models/random_forest_base.joblib')

def format_features(brewery_name:str, review_aroma:int, review_appearance:int, review_palate:int, review_taste:int):
  return {
        'brewery_name': [brewery_name],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste]
    }

@app.get("/")
def read_root():
    return 'Project Objectives: put something here'

@app.get("/health", status_code=200)
def healthcheck():
    return 'Welcome to predictor API, the endpoint is ready'

@app.post("/beer/type")
def predict_beer(brewery_name:str, review_aroma:int, review_appearance:int, review_palate:int, review_taste:int):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    pred = rf_pipe.predict(obs)
    return JSONResponse(pred.tolist())

@app.post("/beers/type")
def predict_beers(df):
    pred = rf_pipe.predict(df)
    return JSONResponse(pred.tolist())

# @app.get("/model/architecture/")
# def get_architecture():
#     return rf_pipe