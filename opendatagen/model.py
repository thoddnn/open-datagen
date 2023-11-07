# model.py
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import numpy as np
import os 

N_RETRIES = 3

class ModelName(Enum):
    GPT_35_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    TEXT_DAVINCI_INSTRUCT = "text-davinci-003"
    GPT_35_TURBO_CHAT = "gpt-3.5-turbo-1106"
    GPT_35_TURBO_16K_CHAT = "gpt-3.5-turbo-16k"
    GPT_4_CHAT = "gpt-4"
    GPT_4_TURBO_CHAT = "gpt-4-1106-preview"

class OpenAIModel:

    client = OpenAI()
    
    def __init__(self, model_name: ModelName):
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name.value

    #@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask_chat_gpt(self, system_prompt:str, user_prompt:str, max_tokens:int, temperature:int, json_mode=False) -> str: 

        if json_mode:

            completion = self.client.chat.completions.create(
            model=self.model_name,
            response_format= {"type": "json_object"},
            messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
            max_tokens=max_tokens,
            temperature=temperature
        )
            
        else:
            
            completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
            max_tokens=max_tokens,
            temperature=temperature
        )


        answer = completion.choices[0].message.content
        
        return answer

    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask_instruct_gpt(self, prompt:str, temperature:int, max_tokens:int, json_mode=False) -> str:

        if json_mode:

            completion = self.client.completions.create(
            model=self.model_name,
            response_format= {"type": "json_object"},
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
            )

        else:

            completion = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
            )

        answer = completion.choices[0].text
        
        return answer

    def create_embedding(self, prompt:str):
        embedding = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=prompt
        )
        return embedding["data"][0]["embedding"]

    def cosine_similarity(self, A:list, B:list):
        return np.dot(A, B)