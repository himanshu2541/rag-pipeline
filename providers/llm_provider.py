from config import Config
from langchain_openai import ChatOpenAI

class LLMProvider:
    """
    Responsible for configuring and providing the LLM instance.
    """
    def __init__(self, config: Config):
        self.config = config

    def get_llm(self):
        """
        Initializes and returns the ChatOpenAI model.
        """
        print("Initializing LLM...")
        llm = ChatOpenAI(
            model=self.config.LLM_MODEL_NAME,
            base_url=self.config.LLM_BASE_URL,
            api_key= lambda: self.config.LLM_API_KEY,
        )
        return llm