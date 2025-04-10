from langchain_ollama import OllamaLLM
import re
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod

# Base system prompt (can be overridden via config)
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. When the context supports an answer, ensure your response is clear, concise, and directly addresses the question.
5. When there is no context, just say you have no context and stop immediately.
6. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
7. Avoid explaining why you cannot answer or speculating about missing details. Simply state that you lack sufficient context when necessary.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Do not mention what you received in context, just focus on answering based on the context.
7. Do not showcase anywhere in the response that you have a supporting text or context for your answers

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Invoke the LLM with the given prompt and return the response."""
        pass

class OllamaLLMClient(BaseLLMClient):
    """Ollama LLM client implementation."""
    
    def __init__(self, model: str):
        self.client = OllamaLLM(model=model, base_url="http://164.52.205.195:11434/")
    
    def invoke(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        # Clean up response by removing <think> tags
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        return response

class LLMConfig:
    """Configuration class for LLM settings."""
    
    def __init__(
        self,
        models: List[Dict[str, str]] = [{"name": "deepseek-r1:14b", "type": "ollama"}],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        default_model: str = "deepseek-r1:14b",
        aggregate_responses: bool = False
    ):
        """
        Args:
            models (List[Dict[str, str]]): List of model configurations with 'name' and 'type'.
            system_prompt (str): The system prompt to use for all LLMs.
            default_model (str): The default model to use if not specified.
            aggregate_responses (bool): Whether to aggregate responses from multiple LLMs.
        """
        self.models = models
        self.system_prompt = system_prompt
        self.default_model = default_model
        self.aggregate_responses = aggregate_responses

class LLMManager:
    """Manages multiple LLM clients and handles invocation."""

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLMManager with a configuration.

        Args:
            config (LLMConfig): Configuration object for LLM settings.
        """
        self.config = config
        self.clients: Dict[str, BaseLLMClient] = self._initialize_clients()

    def _initialize_clients(self) -> Dict[str, BaseLLMClient]:
        """Initialize LLM clients based on the configuration."""
        clients = {}
        for model_config in self.config.models:
            model_name = model_config["name"]
            model_type = model_config["type"].lower()
            
            if model_type == "ollama":
                clients[model_name] = OllamaLLMClient(model=model_name)
            # Add support for other LLM types here in the future, e.g.:
            # elif model_type == "openai":
            #     clients[model_name] = OpenAILLMClient(model=model_name, api_key=...)
            else:
                raise ValueError(f"Unsupported LLM type: {model_type}")
        
        return clients

    def _format_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Format the prompt with system prompt and context if provided."""
        if context:
            return f"{self.config.system_prompt}\n\nContext: {context}\nQuestion: {prompt}"
        return prompt

    def call_llm(self, prompt: str, context: Optional[str] = None, model: Optional[str] = None) -> Union[str, List[str]]:
        """
        Call the LLM(s) with the given prompt and optional context.

        Args:
            prompt (str): The user prompt/question to send to the LLM.
            context (Optional[str]): Additional context to provide to the LLM.
            model (Optional[str]): Specific model to use; defaults to config's default_model.

        Returns:
            Union[str, List[str]]: Single response or list of responses if aggregating.
        """
        model = model or self.config.default_model
        final_prompt = self._format_prompt(prompt, context)

        if self.config.aggregate_responses:
            responses = []
            for model_name, client in self.clients.items():
                response = client.invoke(final_prompt)
                responses.append(response)
            return responses
        else:
            if model not in self.clients:
                raise ValueError(f"Model '{model}' not found in configured clients.")
            return self.clients[model].invoke(final_prompt)

# Example usage
if __name__ == "__main__":
    # Define a configuration
    config = LLMConfig(
        models=[
            {"name": "deepseek-r1:14b", "type": "ollama"},
            # Add more models here in the future, e.g.:
            # {"name": "llama-13b", "type": "ollama"},
        ],
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        default_model="deepseek-r1:14b",
        aggregate_responses=False  # Set to True to get responses from all models
    )

    # Initialize the LLM manager
    llm_manager = LLMManager(config)

    # Call the LLM
    response = llm_manager.call_llm("What is the capital of France?", context="France is a country in Europe.")
    print(response)

    # Example with a specific model
    response = llm_manager.call_llm("What is the weather like?", model="deepseek-r1:14b")
    print(response)
    