"""
LLM module for generating responses using OpenAI API or Ollama.
"""
import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str, context: str = "") -> str:
        """Generate a response given a prompt and context."""
        pass
    
    @abstractmethod
    async def describe_image(self, image_data: bytes, prompt: str = "") -> str:
        """Generate a description for an image."""
        pass


class OpenAIClient(BaseLLM):
    """OpenAI API client for text generation and image description."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        vision_model: str = "gpt-4o-mini",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model to use for text generation.
            vision_model: Model to use for image description.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        """
        from openai import AsyncOpenAI
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.vision_model = vision_model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    async def generate(self, prompt: str, context: str = "") -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            prompt: The user's question.
            context: Retrieved context from RAG.
            
        Returns:
            Generated response text.
        """
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain relevant information, say so honestly.
Keep your answers concise but informative."""

        if context:
            user_message = f"""Context:
{context}

Question: {prompt}

Please answer the question based on the context provided above."""
        else:
            user_message = prompt
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    async def describe_image(self, image_data: bytes, prompt: str = "") -> str:
        """
        Generate a description for an image using OpenAI Vision.
        
        Args:
            image_data: Image file bytes.
            prompt: Optional additional prompt for the description.
            
        Returns:
            Image description text.
        """
        import base64
        
        base64_image = base64.b64encode(image_data).decode("utf-8")
        
        user_prompt = prompt or "Please describe this image in detail."
        
        response = await self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content


class OllamaClient(BaseLLM):
    """Ollama client for local LLM inference."""
    
    def __init__(
        self,
        model: str = "mistral",
        vision_model: str = "llava",
        host: str = "http://localhost:11434"
    ):
        """
        Initialize the Ollama client.
        
        Args:
            model: Model name for text generation.
            vision_model: Model name for image description.
            host: Ollama server URL.
        """
        import ollama
        
        self.model = model
        self.vision_model = vision_model
        self.host = host
        self.client = ollama.AsyncClient(host=host)
    
    async def generate(self, prompt: str, context: str = "") -> str:
        """
        Generate a response using Ollama.
        
        Args:
            prompt: The user's question.
            context: Retrieved context from RAG.
            
        Returns:
            Generated response text.
        """
        if context:
            full_prompt = f"""Context:
{context}

Question: {prompt}

Please answer the question based on the context provided above. Be concise but informative."""
        else:
            full_prompt = prompt
        
        response = await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response["message"]["content"]
    
    async def describe_image(self, image_data: bytes, prompt: str = "") -> str:
        """
        Generate a description for an image using Ollama vision model.
        
        Args:
            image_data: Image file bytes.
            prompt: Optional additional prompt for the description.
            
        Returns:
            Image description text.
        """
        user_prompt = prompt or "Please describe this image in detail."
        
        # Pass raw bytes directly - Ollama library handles encoding
        response = await self.client.chat(
            model=self.vision_model,
            messages=[{
                "role": "user",
                "content": user_prompt,
                "images": [image_data]
            }]
        )
        
        return response["message"]["content"]


class LLMClient:
    """Factory class for creating LLM clients."""
    
    def __init__(
        self,
        provider: str = "openai",
        **kwargs
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider ('openai' or 'ollama').
            **kwargs: Additional arguments passed to the specific client.
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self._client = OpenAIClient(**kwargs)
        elif self.provider == "ollama":
            self._client = OllamaClient(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'ollama'.")
    
    async def generate(self, prompt: str, context: str = "") -> str:
        """Generate a response."""
        return await self._client.generate(prompt, context)
    
    async def describe_image(self, image_data: bytes, prompt: str = "") -> str:
        """Describe an image."""
        return await self._client.describe_image(image_data, prompt)



