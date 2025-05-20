import streamlit as st # type: ignore
import pandas as pd
import utils  # Assuming utils.py contains your data processing functions

# Load data (ensure CSVs are local and included in .gitignore)
df = pd.read_csv('data/merged_country_file.csv')

# Title
st.title("Interactive Dashboard")

# Country Selection Widget
countries = df['country'].unique()
selected_country = st.selectbox("Select a Country:", countries)

# Filter data based on selection
filtered_data = df[df['country'] == selected_country]

# Boxplot of GHI (or other metrics)
st.subheader("Global Horizontal Irradiance (GHI)")
st.box_chart(filtered_data['GHI'])

# Top Regions Table
st.subheader("Top Regions")
top_regions = filtered_data.nlargest(10, 'value_column')  # Replace 'value_column' with actual metric
st.dataframe(top_regions[['region', 'value_column']])