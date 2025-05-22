# https://open-meteo.com/en/docs/historical-forecast-api?start_date=2022-03-01&end_date=2023-02-01&timezone=Europe%2FBerlin&hourly=temperature_2m,relative_humidity_2m

### NOTE ###
# After downlading the data, I delete a couple of rows in the CSV manually
# To get exactly 2022-03-01 00:00:00+00:00 to 

### NOTE ###

import openmeteo_requests
import os
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
	"latitude": 52.52,
	"longitude": 13.41,
	"start_date": "2022-02-28",
	"end_date": "2023-02-01",
	"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_speed_80m", "cloud_cover", "wind_direction_80m", "wind_direction_10m", "weather_code"],
	"timezone": "Europe/Berlin"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
hourly_wind_speed_80m = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_direction_80m = hourly.Variables(6).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(7).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(8).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_speed_80m"] = hourly_wind_speed_80m
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["wind_direction_80m"] = hourly_wind_direction_80m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["weather_code"] = hourly_weather_code

hourly_dataframe = pd.DataFrame(data = hourly_data)

# Create the data/weather directory if it doesn't exist
os.makedirs('data/weather', exist_ok=True)

# Define the output file path
output_file = 'data/weather/hourly_weather_2022_2023.csv'

# Save the DataFrame to CSV
hourly_dataframe.to_csv(output_file, index=False)

print(f"Weather data saved to {output_file}")
