from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import numpy as np
import os 
import json 

from utils import extract_content_from_internet


N_RETRIES = 3

class ModelName(Enum):
    GPT_35_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    TEXT_DAVINCI_INSTRUCT = "text-davinci-003"
    GPT_35_TURBO_CHAT = "gpt-3.5-turbo-1106"
    GPT_35_TURBO_16K_CHAT = "gpt-3.5-turbo-16k"
    GPT_4_CHAT = "gpt-4"
    GPT_4_TURBO_CHAT = "gpt-4-1106-preview"
    TEXT_EMBEDDING_ADA = "text-embedding-ada-002"

class OpenAIChatModel():

    tools = []

    client = OpenAI()

    def __init__(self, model_name: str):
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

    #@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self, max_tokens:int, temperature:int, messages:list, json_mode=False, seed:int =None, use_tools:bool=False) -> str: 
        
        param = {
            
            "model":self.model_name,
            "temperature": temperature,
            "messages": messages,
            
        }
        
        if use_tools:
            param["functions"] = self.tools
        else:
            param["max_tokens"] = max_tokens
        
        if json_mode:
            param["response_format"] = {"type": "json_object"}

        if seed:
            param["seed"] = seed


        completion = self.client.chat.completions.create(**param)

        #if completion.choices[0].finish_reason == "stop":
        
        answer = completion.choices[0].message.content

        return answer 

        #elif completion.choices[0].finish_reason == "length":
        #completion.choices[0].finish_reason == "function_call":

class OpenAIInstructModel():

    client = OpenAI()

    def __init__(self, model_name: ModelName):
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name.value

    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self, prompt:str, temperature:int, max_tokens:int, json_mode=False) -> str:
    
        param = {
            
            "model":self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt": prompt

        }

        if json_mode:
            param["response_format"] = {"type": "json_object"}

 
        completion = self.client.completions.create(**param)

        answer = completion.choices[0].text
        
        return answer

class OpenAIEmbeddingModel():

    client = OpenAI()
    
    def __init__(self, model_name: ModelName):
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name.value

    def create_embedding(self, prompt:str):
        embedding = self.client.embeddings.create(
            model=self.model_name,
            input=prompt
        )

        return embedding["data"][0]["embedding"]

