from agno import Agent as AgnoAgent
from dotenv import load_dotenv

class AgnoAgentWrapper:
    def __init__(self, model = None, api_key = None):
        if model is None:
            model = "groq:llama-3.3-70b-versatile"
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
            
