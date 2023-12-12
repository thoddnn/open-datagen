from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result
from openai import OpenAI
import numpy as np
import os
import json
from opendatagen.utils import is_retryable_answer
import requests
from pydantic import BaseModel, validator, ValidationError, ConfigDict
from typing import Optional, List, Dict, Union, Type
import random 
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

N_RETRIES = 2

class ModelName(Enum):
    GPT_35_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    TEXT_DAVINCI_INSTRUCT = "text-davinci-003"
    GPT_35_TURBO_CHAT = "gpt-3.5-turbo-1106"
    GPT_35_TURBO_16K_CHAT = "gpt-3.5-turbo-16k"
    GPT_4_CHAT = "gpt-4"
    GPT_4_TURBO_CHAT = "gpt-4-1106-preview"
    TEXT_EMBEDDING_ADA = "text-embedding-ada-002"
    SMARTCHUNK = "SmartChunk-0.1-Mistral-7B"
    MISTRAL_7B = "Mistral-7B-v0.1"
    LLAMA_7B = "Llama-2-7b-chat-hf"
    LLAMA_13B = "Llama-2-13b-chat-hf"
    LLAMA_70B = "Llama-2-70b-chat-hf"


class MistralChatModel(BaseModel):

    name:str = "mistral-tiny"
    max_tokens:Optional[int] = 256
    temperature:Optional[float] = 0.7
    messages:Optional[str] = None 
    random_seed:Optional[int] = None 
    top_p:Optional[int] = 1 
    safe_mode:Optional[bool] = False 
    client:Optional[Type[MistralClient]] = None 

    def __init__(self, **data):

        super().__init__(**data)
        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = MistralClient(api_key=api_key)
    

    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self, messages) -> str:
                             
        param = {

            "model":self.name,
            "temperature": self.temperature,
            "messages": messages

        }

        if self.max_tokens:
            param["max_tokens"] = self.max_tokens

        if self.top_p:
            param["top_p"] = self.top_p

        if self.random_seed:
            param["random_seed"] = self.random_seed

        chat_response = self.client.chat(**param)

        answer = chat_response.choices[0].message.content

        return answer


class HuggingFaceModel(BaseModel):

    model_name:str

    def __init__(self, **data):
        super().__init__(**data)

        self.api_token = os.getenv("HUGGINGFACE_API_KEY")
        
    def ask(self, prompt:str):

        headers = {"Authorization": f"Bearer {self.api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
        response = requests.post(API_URL, headers=headers, json=prompt)

        return response.json()

class OpenAIChatModel(BaseModel):

    name:str = "gpt-3.5-turbo-1106"
    system_prompt:Optional[str] = "No verbose."
    max_tokens:Optional[int] = 256
    temperature:Optional[float] = 1
    json_mode:Optional[bool] = False 
    seed:Optional[int] = None 
    tools:Optional[list] = None 
    top_p:Optional[int] = 1 
    stop:Optional[str] = None 
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0 
    client:Optional[Type[OpenAI]] = None 
    
    def __init__(self, **data):
        super().__init__(**data)
        
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")

   
    @retry(retry=retry_if_result(is_retryable_answer), stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self, messages) -> str:
        
        param = {

            "model":self.name,
            "temperature": self.temperature,
            "messages": messages,

        }

        if self.tools:
            param["functions"] = self.tools
        
        if self.max_tokens:
            param["max_tokens"] = self.max_tokens

        if self.seed:
            param["seed"] = self.seed
        
        if self.max_tokens:
            param["max_tokens"] = self.max_tokens

        if self.json_mode:
            param["response_format"] = {"type": "json_object"}

        if self.seed:
            param["seed"] = self.seed

        completion = self.client.chat.completions.create(**param)

        answer = completion.choices[0].message.content

        return answer

class OpenAIInstructModel(BaseModel):

    name:str = "gpt-3.5-turbo-instruct"
    max_tokens:Optional[int] = 256
    temperature:Optional[float] = 1
    messages:Optional[str] = None 
    seed:Optional[int] = None 
    tools:Optional[List[str]] = None 
    start_with:Optional[List[str]] = None
    top_p:Optional[int] = 1 
    stop:Optional[str] = None 
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0 
    client:Optional[Type[OpenAI]] = None 

    def __init__(self, **data):
        super().__init__(**data)

        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")

    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self, messages:str) -> str:
        
        starter = random.choice(self.start_with)
                               
        param = {

            "model":self.name,
            "temperature": self.temperature,
            "prompt": f"{messages}\n\n{starter}"

        }

        if self.tools:
            param["functions"] = self.tools
        
        if self.max_tokens:
            param["max_tokens"] = self.max_tokens

        if self.seed:
            param["seed"] = self.seed

        completion = self.client.completions.create(**param)

        answer = completion.choices[0].text 

        return answer

class OpenAIEmbeddingModel(BaseModel):

    name:str
    
    def __init__(self, **data):
        super().__init__(**data)

        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")

    def create_embedding(self, prompt:str):
        
        embedding = self.client.embeddings.create(
            model=self.model_name,
            input=prompt
        )

        return embedding["data"][0]["embedding"]

class Model(BaseModel):

    openai_chat_model: Optional[OpenAIChatModel] = None 
    huggingface_model:Optional[HuggingFaceModel] = None 
    openai_instruct_model: Optional[OpenAIInstructModel] = None 
    mistral_chat_model:Optional[MistralChatModel] = None 

    def get_model(self):
        if self.openai_chat_model is not None:
            return self.openai_chat_model
        elif self.openai_instruct_model is not None:
            return self.openai_instruct_model
        elif self.huggingface_model is not None:
            return self.huggingface_model
        elif self.mistral_chat_model is not None:
            return self.mistral_chat_model
        else:
            return None