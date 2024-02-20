from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result
from openai import OpenAI
import numpy as np
import os
import json
from opendatagen.utils import is_retryable_answer, pydantic_list_to_dict, load_file
import requests
from pydantic import BaseModel, validator, ValidationError, ConfigDict, Extra
from typing import Optional, List, Dict, Union, Type
import random 
from mistralai.client import MistralClient, ChatMessage
import math 
import tiktoken
from llama_cpp import Llama
import whisper
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, generate, play
import uuid

N_RETRIES = 2

class EvolMethod(Enum):

    deep = "deep"
    concretizing = "concretizing"
    step_reasoning = "step_reasoning"
    breath = "breath"
    basic = "basic"

class UserMessage(BaseModel):

    role:str
    content:str 
    rephraser:Optional[List[EvolMethod]] = None 
    
    def rephrase(self):

        rephraser_name = random.choice(self.rephraser)
        
        prompt = load_file(path=f"files/{rephraser_name}.txt")

        d = {"prompt": self.content}

        prompt = prompt.format(**d)

        model = OpenAIInstructModel(user_prompt=prompt, temperature=[1])

        rephrased_prompt = model.ask()

        self.content = rephrased_prompt

    
    class Config:
        extra = 'forbid'



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

class WhisperModel(BaseModel):

    path:str
    name:Optional[str] = "base"

    class Config:
        extra = 'forbid'

    def ask(self) -> str:

        model = whisper.load_model(self.name)
        result = model.transcribe(self.path)

        text = result["text"]

        return text

class ElevenLabsTTSModel(BaseModel):

    name:Optional[str] = "21m00Tcm4TlvDq8ikWAM"
    user_prompt:str 
    client:Optional[Type[ElevenLabs]] = None 
    
    def ask(self) -> str:

        audio = generate(
            text=self.user_prompt,
            voice=Voice(
                voice_id=self.name,
                settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
                )
        )

        # Generate a random UUID and create a filename
        filename = f'audio_{uuid.uuid4()}.mp3'

        # Save the audio data to a file with the random filename
        with open(filename, 'wb') as audio_file:
            audio_file.write(audio)

        return filename
    
    def __init__(self, **data):

        super().__init__(**data)
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


    class Config:
        extra = 'forbid'


class LlamaCPPModel(BaseModel):

    path:str
    name:Optional[str] = None 
    user_prompt:Optional[str] = None 
    temperature:Optional[List[float]] = [0.8]
    use_gpu:Optional[bool] = False 
    handle_prompt_format:Optional[bool] = False 
    stop:Optional[List[str]] = None  
    max_tokens:Optional[int] = 256
    top_p:Optional[float] = 0.95
    min_p:Optional[float] = 0.05
    echo:Optional[bool] = False
    start_with:Optional[List[str]] = None
    confidence_score:Optional[float] = None

    def ask(self) -> str:

        param_llm = {
            "verbose": False
        }

        if self.use_gpu:
            param_llm["n_gpu_layers"] = -1

        #llm = Llama(model_path=self.path, verbose=False, n_gpu_layers=-1)
        llm = Llama(model_path=self.path, **param_llm)
        
        param_completion = {
            "prompt": f"{self.user_prompt}",
            "max_tokens": self.max_tokens,
            "echo": self.echo,
            "temperature": random.choice(self.temperature)
        }

        if self.stop: 
            param_completion["stop"] = self.stop

        if self.top_p:
            param_completion["top_p"] = self.top_p

        if self.min_p:
            param_completion["min_p"] = self.min_p

        output = llm(**param_completion)

        return output["choices"][0]["text"]
    
    def __init__(self, **data):

        super().__init__(**data)
        self.name = self.path.split('/')[-1]

    class Config:
        extra = 'forbid'



class AnyscaleChatModel(BaseModel):

    name:str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    user_prompt:Optional[List[UserMessage]] = None 
    max_tokens:Optional[int] = 256
    temperature:Optional[List[float]] = [1]
    handle_prompt_format:Optional[bool] = False 
    json_mode:Optional[bool] = False 
    json_schema:Optional[Dict] = None 
    seed:Optional[int] = None 
    tools:Optional[list] = None 
    top_p:Optional[float] = 1 
    stop:Optional[List[str]] = ["</s>", "[/INST]"] 
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0 
    client:Optional[Type[OpenAI]] = None 
    logprobs:Optional[bool] = False 
    confidence_score:Optional[float] = None
    
    def __init__(self, **data):

        super().__init__(**data)
        self.client = OpenAI(api_key=os.getenv("ANYSCALE_API_KEY"), base_url='https://api.endpoints.anyscale.com/v1',)

    class Config:
        extra = 'forbid'


    @retry(retry=retry_if_result(is_retryable_answer), stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self) -> str:
        
        messages = pydantic_list_to_dict(lst = self.user_prompt, fields=['role', 'content']) 

        param = {

            "model":self.name,
            "temperature": random.choice(self.temperature),
            "messages": messages

        }

        if self.stop:
            param["stop"] = self.stop

        if self.top_p:
            param["top_p"] = self.top_p
        
        if self.max_tokens:
            param["max_tokens"] = self.max_tokens

        if self.json_mode and self.json_schema:
            param["response_format"] = {"type": "json_object", "schema": self.json_schema}

        completion = self.client.chat.completions.create(**param)
        
        if self.logprobs:
            self.confidence_score = get_confidence_score(completion=completion)

        answer = completion.choices[0].message.content
        
        return answer

class TogetherChatModel(BaseModel):

    name:str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    user_prompt:Optional[List[UserMessage]] = None 
    max_tokens:Optional[int] = 256
    temperature:Optional[List[float]] = [1]
    json_mode:Optional[bool] = False 
    seed:Optional[int] = None 
    tools:Optional[list] = None 
    top_p:Optional[float] = 1 
    stop:Optional[List[str]] = ["</s>", "[/INST]"] 
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0 
    client:Optional[Type[OpenAI]] = None 
    logprobs:Optional[bool] = False 
    confidence_score:Optional[float] = None
    
    def __init__(self, **data):

        super().__init__(**data)
        self.client = OpenAI(api_key=os.getenv("TOGETHER_API_KEY"), base_url='https://api.together.xyz',)

    class Config:
        extra = 'forbid'


    @retry(retry=retry_if_result(is_retryable_answer), stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self) -> str:

        messages = pydantic_list_to_dict(lst = self.user_prompt, fields=['role', 'content']) 
        
        param = {

            "model":self.name,
            "temperature": random.choice(self.temperature),
            "messages": messages

        }

        if self.stop:
            param["stop"] = self.stop
        
        if self.top_p:
            param["top_p"] = self.top_p
        
        if self.max_tokens:
            param["max_tokens"] = self.max_tokens

        if self.json_mode:
            param["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**param)

        if self.logprobs:
            self.confidence_score = get_confidence_score(completion=completion)

        answer = completion.choices[0].message.content
        
        return answer

class TogetherInstructModel(BaseModel):

    name:str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    max_tokens:Optional[int] = 256
    temperature:Optional[List[float]] = [1]
    handle_prompt_format:Optional[bool] = False 
    user_prompt:Optional[str] = None 
    seed:Optional[int] = None 
    tools:Optional[List[str]] = None 
    start_with:Optional[List[str]] = None
    top_p:Optional[float] = 1 
    stop:Optional[List[str]] = None 
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0 
    client:Optional[Type[OpenAI]] = None 
    confidence_score:Optional[float] = None 

    def __init__(self, **data):

        super().__init__(**data)
        self.client = OpenAI(api_key=os.getenv("TOGETHER_API_KEY"), base_url='https://api.together.xyz',)

    class Config:
        extra = 'forbid'

    
    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self) -> str:

        param = {

            "model":self.name,
            "temperature": random.choice(self.temperature),
            "prompt": f"{self.user_prompt}"

        }

        if self.stop:
            param["stop"] = self.stop

        if self.top_p:
            param["top_p"] = self.top_p
        
        if self.max_tokens:
            param["max_tokens"] = self.max_tokens


        completion = self.client.completions.create(**param)

        answer = completion.choices[0].text 

        return answer


class MistralChatModel(BaseModel):

    name:str = "mistral-tiny"
    max_tokens:Optional[int] = 256
    temperature:Optional[List[float]] = [0.7]
    user_prompt:Optional[List[UserMessage]] = None
    random_seed:Optional[int] = None 
    top_p:Optional[float] = 1 
    safe_mode:Optional[bool] = False 
    client:Optional[Type[MistralClient]] = None 
    confidence_score:Optional[float] = None 
    
    def __init__(self, **data):
        
        super().__init__(**data)
        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = MistralClient(api_key=api_key)

    class Config:
        extra = 'forbid'
    
    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self) -> str:

        messages = [ChatMessage(role=msg.role, content=msg.content) for msg in self.user_prompt]
                         
        param = {

            "model":self.name,
            "temperature": random.choice(self.temperature),
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


class OpenAIChatModel(BaseModel):

    name:str = "gpt-3.5-turbo-1106"
    user_prompt:Optional[List[UserMessage]] = None 
    max_tokens:Optional[int] = 256
    temperature:Optional[List[float]] = [1]
    json_mode:Optional[bool] = False 
    seed:Optional[int] = None 
    tools:Optional[list] = None 
    top_p:Optional[float] = 1 
    note:Optional[List[str]] = None 
    stop:Optional[List[str]] = None 
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0 
    client:Optional[Type[OpenAI]] = None 
    logprobs:Optional[bool] = False 
    confidence_score:Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")

    class Config:
        extra = 'forbid'

    
    @retry(retry=retry_if_result(is_retryable_answer), stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self) -> str:

        messages = pydantic_list_to_dict(lst = self.user_prompt, fields=['role', 'content']) 

        param = {

            "model":self.name,
            "temperature": random.choice(self.temperature),
            "messages": messages,
            "logprobs": self.logprobs

        }

        if self.stop:
            param["stop"] = self.stop

        if self.top_p:
            param["top_p"] = self.top_p
        
        if self.tools:
            param["functions"] = self.tools
        
        if self.max_tokens:
            param["max_tokens"] = self.max_tokens

        if self.max_tokens:
            param["max_tokens"] = self.max_tokens

        if self.json_mode:
            param["response_format"] = {"type": "json_object"}

        if self.seed:
            param["seed"] = self.seed
        
        completion = self.client.chat.completions.create(**param)

        if self.logprobs:
            self.confidence_score = get_confidence_score(completion=completion)

        answer = completion.choices[0].message.content

        return answer


class OpenAIInstructModel(BaseModel):

    name:str = "gpt-3.5-turbo-instruct"
    max_tokens:Optional[int] = 256
    temperature:Optional[List[float]] = [1]
    user_prompt:Optional[str] = None 
    seed:Optional[int] = None 
    tools:Optional[List[str]] = None 
    start_with:Optional[List[str]] = None
    top_p:Optional[float] = 1 
    stop:Optional[List[str]] = None 
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0 
    client:Optional[Type[OpenAI]] = None 
    confidence_score:Optional[float] = None

    class Config:
        extra = 'forbid'

    def __init__(self, **data):
        super().__init__(**data)

        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        
    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
    def ask(self) -> str:

        param = {
            
            "model":self.name,
            "temperature": random.choice(self.temperature),
            "prompt": f"{self.user_prompt}"

        }

        if self.stop:
            param["stop"] = self.stop

        if self.top_p:
            param["top_p"] = self.top_p

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

    client:Optional[Type[OpenAI]] = None 
    name:str = "text-embedding-ada-002"

    class Config:
        extra = 'forbid'
    
    def __init__(self, **data):
        super().__init__(**data)

        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")

    def create_embedding(self, prompt:str):
        
        embedding = self.client.embeddings.create(
            model=self.name,
            input=prompt
        )
        
        return embedding.data[0].embedding

class EmbeddingModel(BaseModel):

    openai_embedding_model:Optional[OpenAIEmbeddingModel] = None 
    
    def get_model(self):
        if self.openai_embedding_model is not None:
            return self.openai_embedding_model
        else:
            return None

class Model(BaseModel):

    openai_chat_model: Optional[OpenAIChatModel] = None 
    openai_instruct_model: Optional[OpenAIInstructModel] = None 
    llamacpp_instruct_model: Optional[LlamaCPPModel] = None 
    mistral_chat_model:Optional[MistralChatModel] = None
    together_chat_model:Optional[TogetherChatModel] = None  
    anyscale_chat_model:Optional[AnyscaleChatModel] = None 
    whisper_model:Optional[WhisperModel] = None 
    elevenlabs_tts_model:Optional[ElevenLabsTTSModel] = None  
    
    def get_model(self):
        if self.openai_chat_model is not None:
            return self.openai_chat_model
        elif self.openai_instruct_model is not None:
            return self.openai_instruct_model
        elif self.mistral_chat_model is not None:
            return self.mistral_chat_model
        elif self.llamacpp_instruct_model is not None:
            return self.llamacpp_instruct_model
        elif self.together_chat_model is not None:
            return self.together_chat_model
        elif self.anyscale_chat_model is not None:
            return self.anyscale_chat_model
        elif self.whisper_model is not None:
            return self.whisper_model
        elif self.elevenlabs_tts_model is not None: 
            return self.elevenlabs_tts_model
        else:
            return None
    


def convert_openailogprobs_to_dict(completion):

    result = {}

    for logp in completion.choices[0].logprobs.content:
        
        result[logp.token] = math.exp(logp.logprob)

    return result 


def extract_keyword_from_text(text:str):

    schema = {
        "type": "object",
        "properties": {
            "keywords": {
            "type": "array",
            "items": {
                "type": "string"
            }
            }
        },
        "required": ["keywords"]
    }

    system_prompt = f"Identify and extract all the important keyphrases from the given text return a valid JSON complying with this schema:\n{str(schema)}"

    user_prompt = f"Text:\n'''{text}'''"

    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":user_prompt}
    ]

    model = OpenAIChatModel(user_prompt=messages, temperature=[0], json_mode=True) 

    answer = model.ask()

    return answer 

def get_confidence_score(completion):

    confidence_score = {}
    
    logp_dict = convert_openailogprobs_to_dict(completion=completion)

    extract = extract_keyword_from_text(text=completion.choices[0].message.content)

    extract_json = json.loads(extract)

    if "keywords" in extract_json:
        keywords = extract_json["keywords"]
    else:
        keywords = []

    if len(keywords) == 0:
        keywords = list(logp_dict.keys())

    for keyword in keywords:

        try:

            encoding = tiktoken.get_encoding("cl100k_base")

            list_of_tokens_integers = encoding.encode(keyword)

            tokens = [encoding.decode_single_token_bytes(token).decode('utf-8') for token in list_of_tokens_integers]
            # Initialize the minimum probability as 1 (maximum possible probability)
            min_probability = 1
            
            for token in tokens:
                # Check if token is in the dictionary and update the minimum probability
                if token in logp_dict and logp_dict[token] < min_probability:
                    min_probability = logp_dict[token]

            # Store the minimum probability as the confidence level for the keyword
            confidence_score[keyword] = min_probability
        
        except UnicodeDecodeError as e:
            
            print(f"Error decoding token {token}: {e}")


    min_confidence_score = min(confidence_score.values())

    return min_confidence_score
