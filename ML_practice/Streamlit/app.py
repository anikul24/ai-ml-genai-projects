import streamlit as st
import pandas as pd
import numpy as np


##title of application

st.title("This is small gaggi !!!")

df = pd.DataFrame({

'f1':[1,2,3],
'f2':[10,20,30]

}
)

##datframe display

st.write("Here is dataframe")
st.write(df)


##chart
chart = pd.DataFrame(
    np.random.randn(20,4),columns=['a','b','c','e']
)

st.line_chart(chart)