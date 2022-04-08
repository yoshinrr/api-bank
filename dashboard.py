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
import numpy as np
url = "https://raw.githubusercontent.com/yoshinrr/api-bank/main/X2.csv"
X = pd.read_csv(url)
X = X.set_index("SK_ID_CURR")
shap_values_tree = joblib.load ("shap_values_tree.pkl")
def request_prediction(data):
    headers = CaseInsensitiveDict()
    headers = {"Content-Type": "application/json", "accept":"application/json"}
    url = "https://oc-projet7.herokuapp.com/predict"
    response = requests.post(url,headers=headers,data=data)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def main():
    

    st.title('Prédiction de solvabilité du client')
    client_id = st.sidebar.selectbox('Identifiant du client',
                                 X.index)

    predict_btn = st.button('Prédire')
    data = {"client_id":client_id}
    data = json.dumps(data)
    pred = request_prediction(data)
    ind = np.where(X.index == client_id)[0][0]


    st.write('Ce client est ',pred["prediction"],'et sa probabilité de solvabilité est de {:.1f}%'.format(pred["probabilite"]))
    
    if st.button('Voir les données utilisées pour la prédicition'):
        #webbrowser.open_new_tab(graph_url)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots()
        st.header('Interprétabilité globale')
        st.pyplot(ax=shap.summary_plot(shap_values_tree, feature_names=X.columns))
        #st.pyplot(fig)
        st.header('Interprétabilité individuelle')
        st.pyplot(ax=shap.bar_plot(shap_values_tree[0][ind], feature_names=X.columns, max_display = 10))
    informations = st.sidebar.selectbox(
        'Quelles informations du client souhaitez vous afficher ?',
        #RATIO_INCOME_GOODS #INCOME_PER_FAMILY_MEMBER
        ['Total des revenus et des crédits', 'Dépenses en biens', 'Ration entre revenu et dépenses'])
    st.title("Informations client")

    st.table(X[client_id])

    if informations == 'Total des revenus et des crédits':
        if ind<20:
            st.bar_chart(X.iloc[ind:ind+19][["AMT_INCOME_TOTAL","AMT_CREDIT"]])
        elif ind>len(X)-20:
            st.bar_chart(X.iloc[ind-19:ind+1][["AMT_INCOME_TOTAL","AMT_CREDIT"]])
        else :
            st.bar_chart(X.iloc[ind-10:ind+10][["AMT_INCOME_TOTAL","AMT_CREDIT"]])
    elif informations == 'Dépenses en biens':
        if ind<20:
            st.bar_chart(X.iloc[ind:ind+19]['AMT_GOODS_PRICE'])
        elif ind>len(X)-20:
            st.bar_chart(X.iloc[ind-19:ind+1]['AMT_GOODS_PRICE'])
        else :
            st.bar_chart(X.iloc[ind-10:ind+10]['AMT_GOODS_PRICE'])
    else:
        if ind<20:
            st.bar_chart(X.iloc[ind:ind+19]['AMT_ANNUITY_x'])
        elif ind>len(X)-20:
            st.bar_chart(X.iloc[ind-19:ind+1]['AMT_ANNUITY_x'])
        else :
            st.bar_chart(X.iloc[ind-10:ind+10]['AMT_ANNUITY_x'])


if __name__ == '__main__':
    main()
