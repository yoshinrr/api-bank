from urllib import response
import pandas as pd
import streamlit as st
import requests
from requests.structures import CaseInsensitiveDict
import json
import webbrowser
import shap
import joblib
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

url = "https://media.githubusercontent.com/media/yoshinrr/api-bank/main/X.csv"
X = pd.read_csv(url)
shap_values_tree = joblib.load ("shap_values_tree.pkl")

def request_prediction(data):
    headers = CaseInsensitiveDict()
    headers = {"Content-Type": "application/json", "accept":"application/json"}
    url1 = "https://oc-projet7.herokuapp.com/predict"
    response = requests.post(url1,headers=headers,data=data)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def main():

    st.title('Prédiction de solvabilité du client')

    client_id = st.sidebar.number_input('Identifiant du client',
                                 min_value=0, max_value=210,
                                value=1, step=1)

    predict_btn = st.button('Prédire')
    data = {"client_id":client_id}
    data = json.dumps(data)
    pred = request_prediction(data)


    st.write('Ce client est ',pred["prediction"],'et sa probabilité de solvabilité est de {:.1f}%'.format(pred["probabilite"]))



    if st.button('Voir les données utilisées pour la prédicition'):
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots()
        st.header('Interprétabilité globale')
        st.pyplot(ax=shap.summary_plot(shap_values_tree, feature_names=X.columns))
        
        st.header('Interprétabilité individuelle')
        st.pyplot(ax=shap.bar_plot(shap_values_tree[0][client_id], feature_names=X.columns, max_display = 10))
        st.title("Informations client")
        
        if client_id<20:
            st.bar_chart(X.iloc[int(client_id):int(client_id)+19][["AMT_INCOME_TOTAL","AMT_CREDIT"]])
        elif client_id>len(X)-20:
            st.bar_chart(X.iloc[int(client_id)-19:int(client_id)+1][["AMT_INCOME_TOTAL","AMT_CREDIT"]])
        else :
            st.bar_chart(X.iloc[int(client_id)-10:int(client_id)+10][["AMT_INCOME_TOTAL","AMT_CREDIT"]])
            
if __name__ == '__main__':
    main()
