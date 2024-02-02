import streamlit as st 
import pickle
import pandas as pd 
import numpy as np 
from PIL import Image


pickle_in = open('artifacts/model.pkl', 'rb')
model = pickle.load(pickle_in)


def predict_water_potability(ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity):

    potability_pred = model.predict([[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]])
    print(potability_pred)
    return potability_pred


def main():
    st.title("Water Potability Check")

    ph = st.text_input('pH level', 'Type here')
    Hardness = st.text_input('Hardness level', 'Type here')
    Solids = st.text_input('Solids measure', 'Type here')
    Chloramines = st.text_input('Chloramines concentration', 'Type here')
    Sulfate = st.text_input('Sulfates concentration', 'Type here')
    Conductivity = st.text_input('Conductivity', 'Type here')
    Organic_carbon = st.text_input('Organic carbon level', 'Type here')
    Trihalomethanes = st.text_input('Trihalomethane concentration', 'Type here')
    Turbidity = st.text_input('Turbidity level', 'Type here')
    result = ""
    if st.button("Predict potability of water sample"):
        result = predict_water_potability(ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity)
        if result == 1:
            st.success("The given water sample is suitable for drinking")
        else:
            st.success("The given water sample is not suitable for drinking")



if __name__ == '__main__':
    main()