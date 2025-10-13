# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from scipy import stats
import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Cache data loading for performance
@st.cache_data
def load_data():
    # Load cleaned CSVs (assuming they are in the 'data/' folder)
    df_benin = pd.read_csv('data/benin_clean.csv', parse_dates=['Timestamp'])
    df_sierraleone = pd.read_csv('data/sierraleone_clean.csv', parse_dates=['Timestamp'])
    df_togo = pd.read_csv('data/togo_clean.csv', parse_dates=['Timestamp'])
    
    # Add country column
    df_benin['country'] = 'Benin'
    df_sierraleone['country'] = 'Sierra Leone'
    df_togo['country'] = 'Togo'
    
    # Concatenate
    df = pd.concat([df_benin, df_sierraleone, df_togo], ignore_index=True)
    
    # Set Timestamp as index
    df.set_index('Timestamp', inplace=True)
    
    return df

# Load the data
df = load_data()

# Function to prepare features and target for a given country
def prepare_data(df, target='GHI', lag_hours=3):
    # Create time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Create lagged features for the target variable
    for i in range(1, lag_hours + 1):
        df[f'{target}_lag_{i}'] = df[target].shift(i)
    
    # Select features
    features = ['Tamb', 'RH', 'WS', 'WSgust', 'WD', 'BP', 'Precipitation', 
                'hour', 'day_of_week', 'month'] + [f'{target}_lag_{i}' for i in range(1, lag_hours + 1)]
    
    # Drop rows with missing values (due to lagging)
    df_clean = df.dropna()
    
    X = df_clean[features]
    y = df_clean[target]
    
    return X, y

# Function to train and evaluate model
@st.cache_data
def train_evaluate_model(X, y, country, target):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, y_test, y_pred, mae, rmse, importance

# Dictionary to store results
countries = ['Benin', 'Sierra Leone', 'Togo']
targets = ['GHI', 'DNI', 'DHI']
results = {country: {} for country in countries}

# Train models for each country and target
for country in countries:
    df_country = df[df['country'] == country]
    for target in targets:
        X, y = prepare_data(df_country, target=target, lag_hours=3)
        if not X.empty and not y.empty:
            model, y_test, y_pred, mae, rmse, importance = train_evaluate_model(X, y, country, target)
            results[country][target] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'y_test': y_test,
                'y_pred': y_pred,
                'importance': importance
            }
            # Print metrics (for debugging in Streamlit)
            st.write(f"\n{country} - {target}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
        else:
            st.write(f"Insufficient data for {target} in {country}")

# Sidebar for filters
st.sidebar.header('Filters')

# Country selection
selected_countries = st.sidebar.multiselect(
    'Select Countries',
    options=countries,
    default=countries
)

# Metric selection
selected_metrics = st.sidebar.multiselect(
    'Select Metrics',
    options=targets,
    default=targets
)

# Date range selection
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date = st.sidebar.date_input('Start Date', min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End Date', max_date, min_value=min_date, max_value=max_date)

# Resampling frequency
freq_options = {'Hourly': 'H', 'Daily': 'D', 'Monthly': 'ME'}
freq_label = st.sidebar.selectbox('Resampling Frequency for Time Series', list(freq_options.keys()), index=1)
resample_freq = freq_options[freq_label]

# Filter the data
if selected_countries and selected_metrics:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
    filtered_df = df[(df.index >= start_dt) & (df.index <= end_dt) & (df['country'].isin(selected_countries))]
else:
    filtered_df = pd.DataFrame()  # Empty if no selection

# Main dashboard
st.title('Solar Data Dashboard - Cross-Country Comparison with Predictive Modeling')

if filtered_df.empty:
    st.warning('Please select at least one country and one metric.')
else:
    # Summary Statistics Table
    st.subheader('Summary Statistics')
    summary = filtered_df.groupby('country')[selected_metrics].agg(['mean', 'median', 'std'])
    st.dataframe(summary.style.format("{:.2f}"))

    # Boxplots for each metric
    st.subheader('Boxplots by Metric')
    for metric in selected_metrics:
        fig_box = px.box(filtered_df, x='country', y=metric, color='country', title=f'Boxplot of {metric} by Country')
        st.plotly_chart(fig_box)

    # Bar Chart for Averages
    st.subheader('Average Metrics by Country')
    avg_df = filtered_df.groupby('country')[selected_metrics].mean().reset_index()
    fig_bar = px.bar(avg_df, x='country', y=selected_metrics, barmode='group', title='Average Metrics by Country')
    st.plotly_chart(fig_bar)

    # Time Series Line Plot (resampled)
    st.subheader('Time Series Analysis')
    if len(selected_countries) > 0:
        resampled_dfs = []
        for country in selected_countries:
            country_df = filtered_df[filtered_df['country'] == country][selected_metrics]
            resampled = country_df.resample(resample_freq).mean()
            resampled['country'] = country
            resampled_dfs.append(resampled)
        resampled_df = pd.concat(resampled_dfs).reset_index()
        
        for metric in selected_metrics:
            fig_line = px.line(resampled_df, x='Timestamp', y=metric, color='country', title=f'{freq_label} Average {metric}')
            st.plotly_chart(fig_line)
    else:
        st.write('No data to display.')

    # Statistical Testing
    st.subheader('Statistical Testing')
    if 'GHI' in selected_metrics and len(selected_countries) >= 2:
        if st.button('Run One-Way ANOVA on GHI'):
            groups = [filtered_df[filtered_df['country'] == c]['GHI'].dropna() for c in selected_countries]
            f_stat, p_val = stats.f_oneway(*groups)
            st.write(f'One-Way ANOVA Results: F-statistic = {f_stat:.4f}, p-value = {p_val:.4e}')
    else:
        st.write('Select at least two countries and include GHI to run ANOVA.')

    # Interactive Forecasting
    st.subheader('Interactive Hourly Forecast')
    if st.button('Generate Forecast for Latest Data'):
        for country in selected_countries:
            df_country = df[df['country'] == country].tail(lag_hours + 1).copy()  # Last lag_hours + 1 rows
            for target in selected_metrics:
                X, _ = prepare_data(df_country, target=target, lag_hours=3)
                if not X.empty:
                    model = results[country][target]['model']
                    pred = model.predict(X)[-1]  # Predict for the last row
                    st.write(f"{country} - Forecasted {target} for next hour: {pred:.2f}")
                else:
                    st.write(f"{country} - Could not forecast {target} due to insufficient data.")

# Summary of Model Performance
st.subheader('Model Performance Summary')
summary_df = pd.DataFrame({
    'Country': [],
    'Metric': [],
    'MAE': [],
    'RMSE': []
})
for country in countries:
    for target in targets:
        if target in results[country]:
            summary_df = pd.concat([summary_df, pd.DataFrame({
                'Country': [country],
                'Metric': [target],
                'MAE': [results[country][target]['mae']],
                'RMSE': [results[country][target]['rmse']]
            })], ignore_index=True)
st.dataframe(summary_df.style.format("{:.2f}"))