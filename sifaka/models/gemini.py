"""
Google Gemini model implementation for Sifaka.

This module provides an implementation of the Model protocol for Google Gemini models.
"""

import os
from typing import Optional, Dict, Any, List

try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, InvalidArgument
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from sifaka.errors import ModelError, ModelAPIError, ConfigurationError


class GeminiModel:
    """Google Gemini model implementation.
    
    This class implements the Model protocol for Google Gemini models.
    
    Attributes:
        model_name: The name of the Gemini model to use.
        api_key: The Google API key to use. If not provided, it will be read from the
            GOOGLE_API_KEY environment variable.
        model: The Gemini model instance.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **options: Any
    ):
        """Initialize the Gemini model.
        
        Args:
            model_name: The name of the Gemini model to use.
            api_key: The Google API key to use. If not provided, it will be read from the
                GOOGLE_API_KEY environment variable.
            **options: Additional options to pass to the Gemini model.
        
        Raises:
            ConfigurationError: If the Google Generative AI package is not installed.
            ModelError: If the API key is not provided and not available in the environment.
        """
        if not GEMINI_AVAILABLE:
            raise ConfigurationError(
                "Google Generative AI package not installed. "
                "Install it with 'pip install google-generativeai'."
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.options = options
        
        if not self.api_key:
            raise ModelError(
                "Google API key not provided. Either pass it as an argument or "
                "set the GOOGLE_API_KEY environment variable."
            )
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Get the model
        try:
            self.model = genai.GenerativeModel(model_name=self.model_name)
        except Exception as e:
            raise ModelError(f"Error initializing Gemini model: {str(e)}")
    
    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the Gemini API.
                Supported options include:
                - temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                  lower values (e.g., 0.2) make it more deterministic.
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - top_k: Controls diversity by limiting to top k tokens.
            
        Returns:
            The generated text.
            
        Raises:
            ModelAPIError: If there is an error communicating with the Gemini API.
        """
        # Merge default options with provided options
        merged_options = {**self.options, **options}
        
        # Convert max_tokens to max_output_tokens if present
        if "max_tokens" in merged_options:
            merged_options["max_output_tokens"] = merged_options.pop("max_tokens")
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=merged_options
            )
            return response.text
        except ResourceExhausted as e:
            raise ModelAPIError(f"Gemini rate limit exceeded: {str(e)}")
        except InvalidArgument as e:
            raise ModelAPIError(f"Invalid argument to Gemini API: {str(e)}")
        except GoogleAPIError as e:
            raise ModelAPIError(f"Gemini API error: {str(e)}")
        except Exception as e:
            raise ModelAPIError(f"Unexpected error when calling Gemini API: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: The text to count tokens in.
            
        Returns:
            The number of tokens in the text.
            
        Raises:
            ModelError: If there is an error counting tokens.
        """
        try:
            # Use Gemini's token counting function
            result = genai.count_tokens(model=self.model_name, prompt=text)
            return result.total_tokens
        except Exception as e:
            raise ModelError(f"Error counting tokens: {str(e)}")
