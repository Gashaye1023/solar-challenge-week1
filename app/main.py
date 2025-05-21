import streamlit as st 
import pandas as pd
import utils  

df = pd.read_csv('../data/merged_country_file.csv')
print("Columns in the DataFrame:", df.columns.tolist())
df.columns = df.columns.str.strip().str.lower()

# Title
st.title("Interactive Dashboard")
# Country Selection Widget
countries = df['country'].unique()
selected_country = st.selectbox("Select a Country:", countries)
filtered_data = df[df['country'] == selected_country]

st.subheader("Global Horizontal Irradiance (GHI)")
st.bar_chart(filtered_data['ghi'].head(10),horizontal=True)

# Top Regions Table
st.subheader("Top Regions by GHI")
top_regions = filtered_data.nlargest(10, 'ghi')
st.table(top_regions[['country', 'ghi']])


# Display filtered data
st.write("Filtered Data for", selected_country)
st.write(filtered_data["ghi"].head(10))  # Display first 10 rows of GHI data