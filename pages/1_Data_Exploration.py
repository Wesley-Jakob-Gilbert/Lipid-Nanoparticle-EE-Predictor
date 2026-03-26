import streamlit as st
import pandas as pd

# Page will show explore the dataset

# import dataframe
df = pd.read_csv("data/lnp_atlas_export.csv")

# Show the actual dataframe
st.write(df)

# Basic analysis of data like Pie chart of lipid types,
# papers, missing data, etc.