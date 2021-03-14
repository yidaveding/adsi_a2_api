from fastapi import FastAPI
from starlette.responses import JSONResponse
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json

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
    def __init__(self, num_features=4, hlayer_neurons=256):
        super(PytorchMultiClass, self).__init__()

        self.layer_1 = nn.Linear(num_features, hlayer_neurons)
        self.layer_2 = nn.Linear(hlayer_neurons, hlayer_neurons)
        self.layer_3 = nn.Linear(hlayer_neurons, hlayer_neurons)
        self.layer_out = nn.Linear(hlayer_neurons, 104)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.dropout(x)
        x = F.relu(self.layer_2(x))
        x = self.dropout(x)
        x = F.relu(self.layer_3(x))
        x = self.dropout(x)
        x = F.relu(self.layer_out(x))
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
    return 'Project Objectives: This is the beer type prediction API. This API use the neural network model to predict beer type based on review scores. Please go to https://secret-sands-74221.herokuapp.com/docs to find out more'

@app.get("/health", status_code=200)
def healthcheck():
    return 'Welcome to predictor API, \n the endpoint is ready'

@app.post("/rf/beer/type")
def predict_beer(brewery_name:str, review_aroma:float, review_appearance:float, review_palate:float, review_taste:float):
    features = format_features(review_aroma, review_appearance, review_palate, review_taste)
    obs = sc.transform(pd.DataFrame(features))
    obst = torch.from_numpy(obs).float()
    pred = nn_model(obst)
    pred = pred.argmax(1)
    pred = enc.inverse_transform(pred)
    return JSONResponse(pred.tolist())

@app.post("/rf/beers/type")
def predict_beers(beer_dict):
    y_pred_list = []
    beer_dict = json.loads(beer_dict)
    for i in beer_dict:
        aroma = beer_dict[i]['review_aroma']
        appearance = beer_dict[i]['review_appearance']
        palate = beer_dict[i]['review_palate']
        taste = beer_dict[i]['review_taste']

        features = format_features(aroma, appearance, palate, taste)
        obs = sc.transform(pd.DataFrame(features))
        obst = torch.from_numpy(obs).float()
        pred = nn_model(obst)
        pred = pred.argmax(1)
        pred = enc.inverse_transform(pred)

        y_pred_list.append(pred.tolist())

        print(y_pred_list)
    return JSONResponse(y_pred_list)

@app.get("/model/architecture/")
def get_architecture():
    mod = nn_model._modules
    arch = {i:mod[i] for i in mod}
    return str(arch)
