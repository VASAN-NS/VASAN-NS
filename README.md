import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import random
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Function to convert text to speech
def speak(text):
    tts = gTTS(text=text, lang='en', slow=False)
    filename = "response.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# Function to listen for audio input
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            print(f"You said: {query}")
            return query.lower()
        except sr.UnknownValueError:
            speak("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            speak("Sorry, my service is currently down.")
            return ""

# Function to get weather information
def get_weather(city):
    api_key = "YOUR_API_KEY"  # Replace with your OpenWeatherMap API key
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"The weather in {city} is currently {weather} with a temperature of {temp}°C."
    else:
        return "I couldn't find that city."

# Function to train a simple anomaly detection model
def train_model():
    data = pd.read_csv('network_traffic.csv')  # Load your dataset
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'cybersecurity_model.pkl')
    print("Model trained and saved.")

# Function to predict anomalies
def predict_anomaly(features):
    model = joblib.load('cybersecurity_model.pkl')
    prediction = model.predict([features])
    return "Anomaly detected!" if prediction[0] == 1 else "No anomalies detected."

# Function to perform calculations
def calculate(expression):
    try:
        result = eval(expression)
        return f"The result is {result}."
    except Exception as e:
        return "There was an error in your calculation."

# Main function to run the assistant
def run_assistant():
    speak("Hello! I'm your personal assistant. How can I help you today?")
    
    # Train the model initially
    train_model()
    
    while True:
        query = listen()
        
        if 'weather' in query:
            city = query.split("in")[-1].strip()
            weather_info = get_weather(city)
            speak(weather_info)
        
        elif 'analyze traffic' in query:
            features_input = input("Enter traffic features separated by commas: ")
            features = [float(x) for x in features_input.split(",")]
            result = predict_anomaly(features)
            speak(result)

        elif 'calculate' in query:
            expression = query.split("calculate")[-1].strip()
            result = calculate(expression)
            speak(result)

        elif 'stop' in query or 'exit' in query:
            speak("Goodbye! Have a great day!")
            break
        
        else:
            responses = [
                "I can help you with the weather, analyze network traffic, or do calculations.",
                "Feel free to ask me anything or say 'stop' to exit."
            ]
            speak(random.choice(responses))

if __name__ == "__main__":
    run_assistant()
