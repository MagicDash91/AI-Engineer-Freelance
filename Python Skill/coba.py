import requests

# Your WeatherStack API key
api_key = "f8bccc8fa4968b7c8baf1360a675a6a3"

# Base URL for WeatherStack API
base_url = "http://api.weatherstack.com/current?"

# Get city name from the user
city_name = input("Enter city name: ")

# Construct the complete URL
complete_url = f"{base_url}access_key={api_key}&query={city_name}"

# Send a GET request to the API
response = requests.get(complete_url)

# Convert the response to JSON format
data = response.json()

# Check if the city is found
if "error" not in data:
    # Extract relevant data from the response
    location = data["location"]
    current_weather = data["current"]
    
    # Get details such as temperature, humidity, and weather description
    city = location["name"]
    country = location["country"]
    temperature = current_weather["temperature"]
    weather_description = current_weather["weather_descriptions"][0]
    humidity = current_weather["humidity"]
    pressure = current_weather["pressure"]
    
    # Print the weather details
    print(f"Weather in {city}, {country}:")
    print(f"Temperature: {temperature}°C")
    print(f"Weather: {weather_description}")
    print(f"Humidity: {humidity}%")
    print(f"Pressure: {pressure} hPa")
else:
    print(f"City {city_name} not found. Please try again.")
