from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result
from openai import OpenAI
import numpy as np
import os
import json
from opendatagen.utils import is_retryable_answer
import requests

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


class HuggingFaceModel():

    def __init__(self, api_token:str, model_id:str):

        self.api_token = os.getenv("HUGGINGFACE_API_KEY")
        self.model_id = model_id

        pass

    def ask(self, prompt:str):

        headers = {"Authorization": f"Bearer {self.api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.post(API_URL, headers=headers, json=prompt)

        return response.json()

class OpenAIChatModel():

    tools = []

    client = OpenAI()

    def __init__(self, model_name: str):
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

    @retry(retry=retry_if_result(is_retryable_answer), stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
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

        answer = completion.choices[0].message.content

        return answer

class OpenAIInstructModel():

    client = OpenAI()

    tools = []

    def __init__(self, model_name: ModelName):
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name.value

    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self, max_tokens:int, temperature:int, messages:list, json_mode=False, seed:int=False, use_tools:bool=False) -> str:

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

        answer = completion.choices[0].message.content

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
