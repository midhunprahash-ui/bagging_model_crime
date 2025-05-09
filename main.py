import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from datetime import datetime, timedelta
import numpy as np

# --- Configuration ---
DATA_FILE_PATH = 'crime_data_sample.csv' # IMPORTANT: Replace with your actual file path
DISTRICT_CENTER_COORDS = [8.7642, 78.1348] # Approx. center of Thoothukudi
DISTRICT_MAP_ZOOM = 11 # Adjust zoom level as needed

# --- Page Configuration (Streamlit) ---
st.set_page_config(
    page_title="Thoothukudi Crime Analysis & Prediction",
    page_icon="🚨",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data # Cache data loading
def load_data(file_path):
    """Loads and preprocesses the crime data."""
    try:
        df = pd.read_csv(file_path)
        # IMPORTANT: Adjust column names if different in your CSV
        required_cols = ['Timestamp', 'AreaName', 'CrimeType', 'Latitude', 'Longitude']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Error: Dataset must contain columns: {', '.join(required_cols)}")
            st.stop()

        # Convert Timestamp
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        except Exception as e:
            st.error(f"Error parsing 'Timestamp' column. Ensure it's in a recognizable date/time format. Details: {e}")
            st.stop()

        # Basic cleaning (optional, customize as needed)
        df.dropna(subset=required_cols, inplace=True) # Drop rows with missing essential data
        df['AreaName'] = df['AreaName'].str.strip().str.title() # Standardize area names
        df['CrimeType'] = df['CrimeType'].str.strip().str.title() # Standardize crime types

        # Extract time features needed later
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek # Monday=0, Sunday=6
        df['Month'] = df['Timestamp'].dt.month
        df['Year'] = df['Timestamp'].dt.year
        df['Date'] = df['Timestamp'].dt.date

        return df

    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        st.info("Please place your 'crime_data.csv' file in the same directory as the script.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        st.stop()

# Define Festival/Event Dates (Customize this extensively!)
# IMPORTANT: Add relevant national, state, and LOCAL Thoothukudi festivals/events
FESTIVAL_DATES = {
    datetime(2023, 1, 14).date(): "Pongal",
    datetime(2023, 1, 15).date(): "Pongal",
    datetime(2023, 1, 16).date(): "Pongal",
    datetime(2023, 4, 14).date(): "Tamil New Year",
    datetime(2023, 8, 15).date(): "Independence Day",
    datetime(2023, 10, 24).date(): "Diwali", # Example dates, update annually
    # Add more dates...
    datetime(2024, 1, 15).date(): "Pongal", # Add dates for the prediction period
    datetime(2024, 1, 16).date(): "Pongal",
    datetime(2024, 1, 17).date(): "Pongal",
    # ... more local events are crucial
}

def is_event_nearby(date, days_buffer=3):
    """Checks if a date is near a known event."""
    for event_date in FESTIVAL_DATES:
        if abs((date - event_date).days) <= days_buffer:
            return 1 # Return 1 if near an event
    return 0 # Return 0 otherwise

# --- Main Application Logic ---
df = load_data(DATA_FILE_PATH)

st.title("🚨 Thoothukudi Crime Analytics & Predictive Deployment System")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Area Crime Insights", "Crime Hotspot Map", "Predictive Deployment Suggestions"])
st.sidebar.markdown("---")
st.sidebar.info(f"Dataset contains {len(df)} records.")
if df is not None and not df.empty:
    st.sidebar.info(f"Date range: {df['Timestamp'].min().strftime('%Y-%m-%d')} to {df['Timestamp'].max().strftime('%Y-%m-%d')}")


# --- Page 1: Area Crime Insights ---
if page == "Area Crime Insights":
    st.header("🔍 Area-Specific Crime Insights")
    st.markdown("Select an area to view its historical crime profile.")

    if df is not None and not df.empty:
        areas = sorted(df['AreaName'].unique())
        selected_area = st.selectbox("Select Area:", areas)

        if selected_area:
            area_df = df[df['AreaName'] == selected_area].copy() # Use .copy() to avoid SettingWithCopyWarning
            st.subheader(f"Analysis for: {selected_area}")

            if area_df.empty:
                st.warning("No crime data found for the selected area in the dataset.")
            else:
                # 1. Top Crime Types
                st.markdown("**Most Frequent Crime Types**")
                top_crimes = area_df['CrimeType'].value_counts().head(10) # Show top 10
                st.bar_chart(top_crimes)
                st.write(top_crimes)

                # 2. Peak Crime Hours
                st.markdown("**Peak Crime Hours**")
                peak_hours = area_df['Hour'].value_counts().sort_index()
                # Ensure all hours 0-23 are present for the chart
                peak_hours = peak_hours.reindex(range(24), fill_value=0)
                st.bar_chart(peak_hours)
                st.write("Crime Count by Hour of Day (0-23):")
                st.write(peak_hours)

                # 3. Recent Trend (Optional: Monthly trend for last year)
                st.markdown("**Monthly Crime Trend (Last 12 Months)**")
                area_df['YearMonth'] = area_df['Timestamp'].dt.to_period('M')
                monthly_trend = area_df[area_df['Timestamp'] > (datetime.now() - timedelta(days=365))]\
                                  .groupby('YearMonth').size()
                if not monthly_trend.empty:
                   # Convert PeriodIndex to string for charting if needed, or use directly if supported
                   monthly_trend.index = monthly_trend.index.strftime('%Y-%m')
                   st.line_chart(monthly_trend)
                else:
                   st.write("Not enough recent data for a monthly trend.")

                # 4. Summary
                if not top_crimes.empty:
                    st.markdown("**Summary:**")
                    top_crime_name = top_crimes.index[0]
                    peak_hour_val = peak_hours.idxmax()
                    st.write(f"Historically, **{selected_area}** has seen the most incidents of **{top_crime_name}**. Crime frequency tends to peak around **{peak_hour_val}:00 - {peak_hour_val+1}:00**.")
    else:
        st.warning("Could not load data to display insights.")

# --- Page 2: Crime Hotspot Map ---
elif page == "Crime Hotspot Map":
    st.header("🗺️ Crime Hotspot Map")
    st.markdown("Visualizing crime concentration across Thoothukudi.")

    if df is not None and not df.empty and 'Latitude' in df.columns and 'Longitude' in df.columns:

        # --- Map Filtering Options ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Map Filters")
        # Date Range Filter
        min_date = df['Timestamp'].min().date()
        max_date = df['Timestamp'].max().date()
        start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.sidebar.error("Error: End date must fall after start date.")
            st.stop()

        # Crime Type Filter
        all_crime_types = ['All'] + sorted(df['CrimeType'].unique())
        selected_crime_types = st.sidebar.multiselect("Filter by Crime Type(s):", all_crime_types, default=['All'])

        # Apply Filters
        map_df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]
        if 'All' not in selected_crime_types and selected_crime_types:
             map_df = map_df[map_df['CrimeType'].isin(selected_crime_types)]

        if map_df.empty:
            st.warning("No crime data matches the selected filters.")
        else:
            st.info(f"Displaying {len(map_df)} crime incidents from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

            # Create Base Map
            m = folium.Map(location=DISTRICT_CENTER_COORDS, zoom_start=DISTRICT_MAP_ZOOM, tiles='CartoDB positron')

            # --- Visualization Layer Options ---
            map_viz_type = st.sidebar.radio("Map Visualization Type", ["HeatMap", "Markers (Clustered)"], index=0)

            if map_viz_type == "HeatMap":
                # Create HeatMap layer
                heat_data = map_df[['Latitude', 'Longitude']].values.tolist()
                HeatMap(heat_data, radius=15).add_to(m)
                st.markdown("**Heatmap:** Shows density of crime incidents. Brighter areas indicate higher concentration.")

            elif map_viz_type == "Markers (Clustered)":
                 # Define colors (Customize based on your major crime types)
                 # IMPORTANT: Update this dictionary with your actual crime types and desired colors
                 crime_color_map = {
                    'Theft': 'orange',
                    'Assault': 'red',
                    'Burglary': 'purple',
                    'Robbery': 'darkred',
                    'Vandalism': 'lightgray',
                    'Drug Offense': 'blue',
                    # Add more mappings...
                    'Other': 'gray' # Default color
                 }

                 marker_cluster = MarkerCluster().add_to(m)
                 for idx, row in map_df.iterrows():
                    crime_type = row['CrimeType']
                    color = crime_color_map.get(crime_type, 'gray') # Get color or default

                    # Create popup text
                    popup_text = f"""
                    <b>Type:</b> {row['CrimeType']}<br>
                    <b>Date:</b> {row['Timestamp'].strftime('%Y-%m-%d %H:%M')}<br>
                    <b>Area:</b> {row['AreaName']}
                    """
                    iframe = folium.IFrame(popup_text, width=200, height=100)
                    popup = folium.Popup(iframe, max_width=200)


                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=popup,
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(marker_cluster)
                 st.markdown("**Clustered Markers:** Shows individual crime incidents grouped by proximity. Colors indicate crime type (see legend below - if you add one). Click clusters to zoom.")
                 # Optional: Add a legend for marker colors if needed


            # Display Map
            st_folium(m, width=1000, height=600)

    else:
        st.warning("Could not load data or missing Latitude/Longitude columns required for mapping.")


# --- Page 3: Predictive Deployment Suggestions ---
elif page == "Predictive Deployment Suggestions":
    st.header("📈 Predictive Deployment Suggestions")
    st.markdown("Forecasts potential high-risk areas for the next month based on historical data and events.")
    st.warning("⚠️ Disclaimer: Predictions are based on historical patterns and known events. They indicate probability, not certainty. Unforeseen events can alter crime patterns. Use as an informational tool.")

    if df is not None and not df.empty:
        try:
            # --- Feature Engineering ---
            st.subheader("Model Training Data Preparation")
            st.write("Preparing data for prediction...")

            # Aggregate data by Date and Area to get daily crime counts
            daily_crimes = df.groupby(['Date', 'AreaName']).size().reset_index(name='CrimeCount')

            # Create a full date range and area combinations to handle days with zero crime
            min_date = df['Date'].min()
            # Use current date for max_date if data is recent, otherwise use max from data
            max_date_data = df['Date'].max()
            current_date = datetime.now().date()
            max_date = max(max_date_data, current_date - timedelta(days=1)) # Predict starting tomorrow

            all_dates = pd.date_range(start=min_date, end=max_date, freq='D').date
            all_areas = df['AreaName'].unique()
            index_multi = pd.MultiIndex.from_product([all_dates, all_areas], names=['Date', 'AreaName'])
            full_df = pd.DataFrame(index=index_multi).reset_index()

            # Merge with actual crime counts
            full_df = pd.merge(full_df, daily_crimes, on=['Date', 'AreaName'], how='left')
            full_df['CrimeCount'].fillna(0, inplace=True) # Fill days with no recorded crime with 0

            # Feature Engineering (Add more features for better accuracy)
            full_df['Timestamp'] = pd.to_datetime(full_df['Date']) # Convert Date back to datetime for dt accessor
            full_df['DayOfWeek'] = full_df['Timestamp'].dt.dayofweek
            full_df['Month'] = full_df['Timestamp'].dt.month
            full_df['DayOfYear'] = full_df['Timestamp'].dt.dayofyear
            full_df['WeekOfYear'] = full_df['Timestamp'].dt.isocalendar().week.astype(int)
            full_df['Year'] = full_df['Timestamp'].dt.year

            # Add event feature
            full_df['IsEventNearby'] = full_df['Date'].apply(lambda x: is_event_nearby(x, days_buffer=3)) # 3 day buffer around events

            # Lag features (simple example: crimes yesterday in the same area)
            # Note: More sophisticated lags (e.g., same day last week) are often better
            full_df.sort_values(by=['AreaName', 'Date'], inplace=True)
            full_df['CrimeCountLag1'] = full_df.groupby('AreaName')['CrimeCount'].shift(1).fillna(0)
            # Add more lags (e.g., lag 7, lag 30) and rolling averages for better results

            # Encode AreaName
            area_encoder = LabelEncoder()
            full_df['AreaNameEncoded'] = area_encoder.fit_transform(full_df['AreaName'])

            # --- Model Training ---
            features = ['AreaNameEncoded', 'DayOfWeek', 'Month', 'DayOfYear', 'WeekOfYear', 'Year', 'IsEventNearby', 'CrimeCountLag1']
            target = 'CrimeCount'

            # Use data up to yesterday for training
            train_df = full_df[full_df['Date'] < current_date].copy()
            train_df = train_df.dropna(subset=features + [target]) # Drop rows where features couldn't be calculated (e.g., first lag day)

            if len(train_df) < 100: # Need sufficient data to train
                 st.error("Insufficient historical data to train the prediction model after processing.")
                 st.stop()


            X = train_df[features]
            y = train_df[target]

            # Train the LightGBM Model (Caching model might be complex with Streamlit's flow, retrain daily/on demand)
            # For production, consider saving/loading model artifacts (e.g., using joblib)
            st.write(f"Training prediction model on {len(X)} data points...")
            model = lgb.LGBMRegressor(objective='poisson', # Good for count data
                                      metric='rmse',
                                      n_estimators=100, # Adjust parameters as needed
                                      learning_rate=0.1,
                                      num_leaves=31,
                                      random_state=42,
                                      n_jobs=-1) # Use all available CPU cores
            model.fit(X, y)
            st.success("Model training complete.")

            # --- Prediction for Next Month ---
            st.subheader("🗓️ Crime Risk Forecast (Next 30 Days)")
            prediction_start_date = current_date
            prediction_end_date = current_date + timedelta(days=30)
            prediction_dates = pd.date_range(start=prediction_start_date, end=prediction_end_date, freq='D').date

            future_df_list = []
            last_known_data = full_df[full_df['Date'] == (current_date - timedelta(days=1))] #[['AreaName', 'AreaNameEncoded', 'CrimeCount']]

            # Prepare feature data for each future day
            for date in prediction_dates:
                daily_future = pd.DataFrame({'AreaNameEncoded': area_encoder.transform(all_areas), 'AreaName': all_areas})
                daily_future['Date'] = date
                daily_future['Timestamp'] = pd.to_datetime(daily_future['Date'])
                daily_future['DayOfWeek'] = daily_future['Timestamp'].dt.dayofweek
                daily_future['Month'] = daily_future['Timestamp'].dt.month
                daily_future['DayOfYear'] = daily_future['Timestamp'].dt.dayofyear
                daily_future['WeekOfYear'] = daily_future['Timestamp'].dt.isocalendar().week.astype(int)
                daily_future['Year'] = daily_future['Timestamp'].dt.year
                daily_future['IsEventNearby'] = daily_future['Date'].apply(lambda x: is_event_nearby(x, days_buffer=3))

                # Estimate Lag Feature (Use prediction from day before, or last known value)
                # This is a simplification. Proper time series forecasting handles this iteratively.
                if date == prediction_start_date:
                    # Use the last actual known lag value
                     lag_map = last_known_data.set_index('AreaNameEncoded')['CrimeCount']
                     daily_future['CrimeCountLag1'] = daily_future['AreaNameEncoded'].map(lag_map).fillna(0)
                else:
                     # Use the *predicted* value from the previous day as the lag for the current day
                     # Requires storing previous day's predictions - more complex state management needed
                     # Simple approach: Use the last *known* lag again (less accurate over time)
                     lag_map = last_known_data.set_index('AreaNameEncoded')['CrimeCount'] # Reusing last known actuals
                     daily_future['CrimeCountLag1'] = daily_future['AreaNameEncoded'].map(lag_map).fillna(0)
                     # *** Improvement needed here for multi-step forecasting ***

                future_df_list.append(daily_future)

            future_df = pd.concat(future_df_list, ignore_index=True)

            # Make Predictions
            future_predictions = model.predict(future_df[features])
            future_df['PredictedCrimeCount'] = np.maximum(0, future_predictions) # Ensure predictions are non-negative

            # --- Deployment Suggestions ---
            st.markdown("**Deployment Focus Areas (Aggregated Risk - Next 30 Days)**")

            # Aggregate risk over the prediction period
            area_risk = future_df.groupby('AreaName')['PredictedCrimeCount'].sum().reset_index()
            area_risk.rename(columns={'PredictedCrimeCount': 'TotalPredictedRisk'}, inplace=True)
            area_risk = area_risk.sort_values(by='TotalPredictedRisk', ascending=False)

            # Get historical peak times for top risk areas
            def get_peak_time_str(area_name):
                area_hist_df = df[df['AreaName'] == area_name]
                if not area_hist_df.empty:
                    peak_hours = area_hist_df['Hour'].value_counts().sort_index()
                    peak_hours = peak_hours.reindex(range(24), fill_value=0)
                    if peak_hours.sum() > 0:
                        peak_start = peak_hours.idxmax()
                        # Suggest a window (e.g., peak hour +/- 1 hour or a 4-hour block)
                        return f"{peak_start:02d}:00 - {(peak_start + 2) % 24:02d}:00" # Example: 2-hour window after peak
                return "N/A"

            area_risk['SuggestedFocusTime'] = area_risk['AreaName'].apply(get_peak_time_str)

            # Define simple deployment levels based on relative risk
            # (Customize thresholds based on operational capacity)
            risk_quantiles = area_risk['TotalPredictedRisk'].quantile([0.75, 0.90]) # Example: Top 25% and Top 10%
            def assign_deployment_level(risk):
                if risk >= risk_quantiles[0.90]:
                    return "High Priority (4+ Officers)"
                elif risk >= risk_quantiles[0.75]:
                    return "Medium Priority (2-4 Officers)"
                else:
                    return "Standard Patrol (1-2 Officers)"

            area_risk['SuggestedDeployment'] = area_risk['TotalPredictedRisk'].apply(assign_deployment_level)

            # Display Suggestions Table
            st.dataframe(area_risk[['AreaName', 'TotalPredictedRisk', 'SuggestedDeployment', 'SuggestedFocusTime']]
                         .head(15), # Show top 15 areas
                         use_container_width=True) # Use full width

            # Highlight areas potentially affected by upcoming events
            st.markdown("**Note on Events:**")
            event_focus_areas = future_df[future_df['IsEventNearby'] == 1]['AreaName'].unique()
            if len(event_focus_areas) > 0:
                 st.write("Consider increased vigilance in the following areas during periods near upcoming events:")
                 st.write(f"- {', '.join(event_focus_areas)}")
            else:
                 st.write("No major events identified affecting predictions in the next 30 days based on the provided list.")


        except Exception as e:
            st.error(f"An error occurred during prediction or suggestion generation: {e}")
            import traceback
            st.text(traceback.format_exc()) # Show detailed error for debugging

    else:
        st.warning("Could not load data for prediction.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Developed for Thoothukudi Police")
st.sidebar.write("Data Science & Analytics Team")
st.sidebar.write("© 2023")
st.sidebar.write("All rights reserved.")
st.sidebar.markdown("**Disclaimer:** This is a prototype system. Predictions are based on historical data and may not reflect real-time conditions.")
st.sidebar.markdown("**Contact:** [Your Contact Info]") # Add your contact info or organization link
st.sidebar.markdown("**Feedback:** [Your Feedback Link]") # Add a feedback link or form
st.sidebar.markdown("**GitHub:** [Your GitHub Link]") # Add your GitHub or project link
st.sidebar.markdown("**License:** [Your License Info]") # Add license info if applicable
# --- End of Script ---
```