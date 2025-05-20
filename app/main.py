import streamlit as st 
import pandas as pd
import utils  

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
# Top Regions Table with Counts
st.subheader("Top Regions")
top_regions = filtered_data['country'].value_counts().nlargest(10).reset_index()
top_regions.columns = ['country', 'count']
# Display the counts
st.dataframe(top_regions)