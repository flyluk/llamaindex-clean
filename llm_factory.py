"""Factory for creating LLM and embedding instances based on configuration."""

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(base_url, model, api_key=None, context_length=32768):
        """Create LLM instance based on configuration."""
        is_azure = "azure" in base_url.lower()
        
        if is_azure:
            from llama_index.llms.azure_openai import AzureOpenAI
            return AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                engine=model,
                api_version="2024-02-01"
            )
        elif api_key:
            from llama_index.llms.openai import OpenAI
            return OpenAI(
                api_key=api_key,
                model=model,
                api_base=base_url
            )
        else:
            return Ollama(
                model=model,
                request_timeout=240.0,
                context_window=context_length,
                base_url=base_url
            )
    
    @staticmethod
    def create_embedding(embed_url, embed_model, api_key=None):
        """Create embedding instance based on configuration."""
        is_azure = "azure" in embed_url.lower()
        is_ollama = not is_azure and ("localhost" in embed_url or "11434" in embed_url)
        
        if is_ollama:
            return OllamaEmbedding(
                model_name=embed_model,
                base_url=embed_url,
                embed_batch_size=1
            )
        elif is_azure:
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
            return AzureOpenAIEmbedding(
                api_key=api_key,
                azure_endpoint=embed_url,
                engine=embed_model,
                api_version="2024-02-01"
            )
        elif api_key:
            from llama_index.embeddings.openai import OpenAIEmbedding
            return OpenAIEmbedding(
                api_key=api_key,
                api_base=embed_url,
                model=embed_model
            )
        else:
            return OllamaEmbedding(
                model_name=embed_model,
                base_url=embed_url,
                embed_batch_size=1
            )
    
    @staticmethod
    def setup_settings(base_url, model, embed_url, embed_model, api_key=None, context_length=32768):
        """Setup global LlamaIndex settings."""
        Settings.llm = LLMFactory.create_llm(base_url, model, api_key, context_length)
        Settings.embed_model = LLMFactory.create_embedding(embed_url, embed_model, api_key)
