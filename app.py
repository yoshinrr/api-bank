import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
from id import get_id
import matplotlib.pyplot as plt
import re
import shap
from io import BytesIO
from starlette.responses import StreamingResponse

app = FastAPI()
loaded_model = joblib.load('model.pkl')
url = "https://media.githubusercontent.com/media/yoshinrr/api-bank/main/X.csv"
X = pd.read_csv(url)
shap_values = joblib.load ("shap_values_tree.pkl")
explainer = shap.TreeExplainer(loaded_model)
#shap_values_tree = explainer.shap_values((X.loc[[client_id]]).values)

@app.get('/')
def index():
    return{"message": "Bienvenue à la banque"}


@app.post('/predict')
def solvability(data: get_id):
    data = data.dict()
    client_id = data['client_id']
   # print(model.predict([client_id, probability]))
    prediction  = loaded_model.predict((X.loc[[client_id]]).values)
    probabilite = loaded_model.predict_proba((X.loc[[client_id]]).values)
    probabilite = probabilite[0][0]*100
    if(prediction[0] > 0.):
        prediction  = "non solvable"
        
    else:
        prediction = "solvable"
        
    return {
        'prediction': prediction,
        'probabilite': probabilite
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

