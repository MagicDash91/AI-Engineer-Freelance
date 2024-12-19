from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import requests
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import logging
import google.generativeai as genai

# WeatherStack API Constants
API_KEY = "*********************************"
BASE_URL = "http://api.weatherstack.com/current"

# Google Gemini API Configuration
GEMINI_API_KEY = "*****************************"
genai.configure(api_key=GEMINI_API_KEY)

# Template renderer
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

class WeatherResponse(BaseModel):
    current_temperature: float
    weather_description: str
    wind_speed: float
    humidity: float

def fetch_lat_lon_from_city(city: str):
    """
    Get latitude and longitude from the city name using WeatherStack API.
    """
    params = {
        "access_key": API_KEY,
        "query": city
    }
    logging.debug(f"Fetching latitude and longitude for city: {city}")
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        logging.error(f"Failed to fetch geolocation data: {response.status_code}")
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch geolocation data")
    
    data = response.json()
    logging.debug(f"Geolocation data response: {data}")

    if "location" not in data:
        logging.error(f"City '{city}' not found in geolocation data")
        raise HTTPException(status_code=404, detail=f"City '{city}' not found or invalid")
    
    lat = data["location"]["lat"]
    lon = data["location"]["lon"]
    logging.debug(f"Found coordinates for {city}: Lat: {lat}, Lon: {lon}")
    return lat, lon

def fetch_weather_data(latitude: float, longitude: float):
    """
    Fetch current weather data from WeatherStack API using latitude and longitude.
    """
    params = {
        "access_key": API_KEY,
        "query": f"{latitude},{longitude}"
    }
    logging.debug(f"Fetching current weather data for coordinates: Lat: {latitude}, Lon: {longitude}")
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        logging.error(f"Failed to fetch weather data: {response.status_code}")
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch weather data")
    
    data = response.json()
    logging.debug(f"Weather data response: {data}")
    return data

def generate_gemini_response(city: str, weather_desc: str, temperature: float):
    """
    Generate a response using Google Gemini, explaining actions based on the weather.
    """
    prompt = f"Explain what a person might do in {city} if the weather is {weather_desc} and the temperature is {temperature}°C."
    logging.debug(f"Generating content for: {prompt}")
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    if response.text:
        logging.debug(f"Gemini response: {response.text}")
        return response.text
    else:
        logging.error("Failed to generate content from Gemini.")
        return "Sorry, I couldn't generate a response."

@app.get("/", response_class=HTMLResponse)
def weather_input_form(request: Request):
    """
    Input form for city name.
    """
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "current_temperature": None,
            "weather_description": None,
            "wind_speed": None,
            "humidity": None,
            "gemini_response": None,
            "error": None
        }
    )

@app.get("/weather", response_class=HTMLResponse)
def get_weather(
    request: Request,
    city: str = Query(..., description="City name")
):
    """
    Endpoint to get current weather and generate Gemini response.
    """
    try:
        # Fetch latitude and longitude based on city name
        latitude, longitude = fetch_lat_lon_from_city(city)
        
        # Fetch current weather data using coordinates
        weather_data = fetch_weather_data(latitude, longitude)
        
        # Extract current weather details
        current_temp = weather_data['current']['temperature']
        weather_desc = weather_data['current']['weather_descriptions'][0]
        wind_speed = weather_data['current']['wind_speed']
        humidity = weather_data['current']['humidity']

        # Generate content from Gemini based on weather report
        gemini_response = generate_gemini_response(city, weather_desc, current_temp)

        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "current_temperature": current_temp,
                "weather_description": weather_desc,
                "wind_speed": wind_speed,
                "humidity": humidity,
                "gemini_response": gemini_response,
                "error": None
            }
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "current_temperature": None,
                "weather_description": None,
                "wind_speed": None,
                "humidity": None,
                "gemini_response": None,
                "error": e.detail
            }
        )
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "current_temperature": None,
                "weather_description": None,
                "wind_speed": None,
                "humidity": None,
                "gemini_response": None,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
