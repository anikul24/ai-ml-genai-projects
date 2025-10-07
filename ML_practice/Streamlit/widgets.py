import streamlit as st
import pandas as pd
import numpy as np

st.title("Text input")

name = st.text_input("Your name ?")

age = st.slider("Select your age:",0,100,5)

options = ["Brynn","Arleigh","Claire","Zaina"]
choice = st.selectbox("Chose your fiend :", options)

if name:
    st.write(f"No, you are not {name} , you are small girl And your age is {age} and your dear friend is {choice}!! ")