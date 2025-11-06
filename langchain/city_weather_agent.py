
import os
from dotenv import load_dotenv
import requests
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

load_dotenv()

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    api_url = f"http://api.weatherapi.com/v1/current.json?key={os.environ.get("WEATHER_API_KEY")}&q={city}&aqi=no"
    response = requests.get(api_url)
    if response.status_code != 200:
        return f"Unable to get Weather forcast for city {city}"
    return response.json()["current"]["condition"]["text"]

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    debug=False
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in London"}]}
)

print(result['messages'][-1].content)
