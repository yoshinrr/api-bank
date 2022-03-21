import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
from id import get_id
import re

app = FastAPI()
loaded_model = joblib.load('model.pkl')
X = pd.read_csv("X.csv")

@app.get('/')
def index():
    return{"message": "Bienvenue à la banque"}


@app.post('/predict')
def solvability(data: get_id):
    data = data.dict()
    client_id = data['client_id']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction  = loaded_model.predict((X.loc[[client_id]]).values)
    probabilite = loaded_model.predict_proba((X.loc[[client_id]]).values)
    if(prediction[0] > 0.):
        prediction  = "Non solvable"
        probabilite = "Probabilité de non solvabilité: ",probabilite[0][1]*100, "%"
    else:
        prediction = "Solvable"
        probabilite = "Probabilité de solvabilité: ",probabilite[0][0]*100, "%"
    return {
        'prediction': prediction,
        'probabilite': probabilite
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn app:app --reload
# 122334
# déployer api sur heroku
# dashboard avec sidebar pour choisir client
# afficher infos client
# avoir un bouton pour prédire (appel d'api)
# afficher prédiction proba + solva
# bouton afficher interprétabilité (sauvegarder et importer shap value)
# placer valeur client sur distribution de 3/4 variables
