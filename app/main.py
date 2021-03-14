from fastapi import FastAPI
from starlette.responses import JSONResponse
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

'''
‘/’ (GET): Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project
‘/health/’ (GET): Returning status code 200 with a string with a welcome message of your choice
‘/beer/type/’ (POST): Returning prediction for a single input only
‘/beers/type/’ (POST): Returning predictions for a multiple inputs
‘/model/architecture/’ (GET): Displaying the architecture of your Neural Networks (listing of all layers with their types)
'''

app = FastAPI()

rf_pipe = joblib.load('../models/random_forest_base.joblib')
sc = joblib.load('../models/standard_scaler.joblib')
enc = joblib.load('../models/output_encoder.joblib')

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()

        self.layer_1 = nn.Linear(num_features, 128)
        self.layer_out = nn.Linear(128, 104)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)

nn_model = PytorchMultiClass(4)
nn_model.load_state_dict(torch.load('../models/pt_hpt_state_dict'))

def format_features(review_aroma:int, review_appearance:int, review_palate:int, review_taste:int):
  return {
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
    return 'Welcome to predictor API, \n the endpoint is ready'

@app.post("/rf/beer/type")
def predict_beer(brewery_name:str, review_aroma:int, review_appearance:int, review_palate:int, review_taste:int):
    features = format_features(review_aroma, review_appearance, review_palate, review_taste)
    obs = sc.transform(pd.DataFrame(features))
    obst = torch.from_numpy(obs).float()
    pred = nn_model(obst)
    pred = pred.argmax(1)
    pred = enc.inverse_transform(pred)
    return JSONResponse(pred.tolist())

@app.post("/rf/beers/type")
def predict_beers(df):
    pred = enc.inverse_transform(nn_model(df))
    return JSONResponse(pred.tolist())

@app.get("/model/architecture/")
def get_architecture():
    mod = nn_model._modules
    arch = {i:mod[i] for i in mod}
    return str(arch)
