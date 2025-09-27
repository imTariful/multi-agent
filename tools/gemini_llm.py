import os
from typing import List, Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions


class _LLMResponse:
    def __init__(self, content: str):
        self.content = content


class GeminiLLM:
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        self._requested_model_name = model_name
        self._fallback_models = [
            model_name,
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
        ]
        self.model = genai.GenerativeModel(self._fallback_models[0])
        self.generation_config = {
            "temperature": temperature,
        }

    def _messages_to_text(self, messages: List[Any]) -> str:
        parts = []
        for m in messages:
            # LangChain messages have .content and type markers, we just join
            content = getattr(m, "content", str(m))
            parts.append(str(content))
        return "\n\n".join(parts)

    def invoke(self, messages: List[Any]):
        prompt = self._messages_to_text(messages)
        last_error = None
        
        # First, try to get available models
        try:
            available_models = genai.list_models()
            model_names = [model.name for model in available_models if 'generateContent' in model.supported_generation_methods]
            
            # Try available models first
            for model_name in model_names:
                if any(fallback in model_name for fallback in self._fallback_models):
                    try:
                        model = genai.GenerativeModel(model_name)
                        resp = model.generate_content(prompt, generation_config=self.generation_config)
                        text = resp.text or ""
                        return _LLMResponse(text)
                    except Exception as e:
                        last_error = e
                        continue
        except Exception as e:
            last_error = e
        
        # Fallback to original method if listing models fails
        for idx, model_name in enumerate(self._fallback_models):
            try:
                if idx > 0:
                    # Recreate model with the fallback name
                    self.model = genai.GenerativeModel(model_name)
                resp = self.model.generate_content(prompt, generation_config=self.generation_config)
                text = resp.text or ""
                return _LLMResponse(text)
            except google_exceptions.NotFound as e:
                last_error = e
                continue
            except Exception as e:
                # For other errors (quota/timeouts), surface immediately
                raise
        
        # If we exhausted fallbacks, raise a helpful message
        raise RuntimeError(
            f"None of the requested Gemini models are accessible with this API key: {self._fallback_models}. "
            f"Original error: {last_error}. "
            f"Please check your API key and ensure it has access to Gemini models."
        )

    def __call__(self, messages: List[Any]):
        return self.invoke(messages)


