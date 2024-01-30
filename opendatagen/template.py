from pydantic import BaseModel, validator, ValidationError, ConfigDict
from typing import Optional, List, Dict, Union, Any , Callable
from enum import Enum
import os
import json
from opendatagen.utils import load_file
from opendatagen.model import OpenAIChatModel, OpenAIInstructModel, LlamaCPPModel, Model, EmbeddingModel, MistralChatModel, AnyscaleChatModel, TogetherChatModel, TogetherInstructModel
from mistralai.models.chat_completion import ChatMessage
from urllib.parse import quote_plus
import requests
import trafilatura
from PyPDF2 import PdfReader
import pandas as pd
from datasets import load_dataset, Dataset
from opendatagen.utils import get_first_n_tokens, num_tokens_from_string, cosine_similarity
import random
import uuid
import re 
import pandas as pd
import numpy as np 

class DeleteMode(Enum):
    HIGHEST = 'highest'
    LOWEST = 'lowest'

class RAGHuggingFace(BaseModel):

    dataset_path:str
    dataset_name:Optional[str] = None
    data_dir:Optional[str] = None
    column_name:str
    specific_row:Optional[int] = None 
    streaming:bool = True
    min_tokens:Optional[int] = 0
    max_tokens:Optional[int] = None
    subset_size:Optional[int] = 10000
    subset:Optional[List[str]] = None   
    dst:Optional[Any] = None 

    class Config:
        extra = "forbid"
    

    def get_random_value_from_dataset(self):

        if self.subset == None: 

            param = {}

            if self.dataset_path:
                param["path"] = self.dataset_path

            if self.data_dir:
                param["data_dir"] = self.data_dir

            if self.dataset_name:
                param["name"] = self.dataset_name

            param["streaming"] = self.streaming

            self.dst = load_dataset(**param)

            self.subset = [sample[self.column_name] for _, sample in zip(range(self.subset_size), self.dst["train"])]

            self.dst = None 

        max_attempts = 50

        if self.specific_row:
            max_attempts = 1
      
        count = 0

        while count < max_attempts:

            if self.specific_row:
                index = self.specific_row
            else:
                index = random.randint(0, len(self.subset) - 1)
                
            text = self.subset[index]

            num_tokens = num_tokens_from_string(text, encoding_name="cl100k_base")

            if num_tokens >= self.min_tokens:

                if self.max_tokens:

                    text = self.subset[index]

                    result = get_first_n_tokens(n=self.max_tokens, text=text, encoding_name="cl100k_base")
                    
                    return result

                else:

                    result = self.subset[index]

                    return result

            count = count + 1


class RAGLocalPath(BaseModel):

    localPath:Optional[str] = None
    directoryPath:Optional[str] = None
    content:Optional[str] = None
    randomize:Optional[bool] = False  
    sample_size: Optional[float] = 0.1

    class Config:
        extra = "forbid"

    def get_random_csv_chunk(self, df: pd.DataFrame):
        # Randomly sample a fraction of the dataframe rows
        return df.sample(frac=self.sample_size)
    
    def get_random_text_chunk(self, text):

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sample_size = max(1, int(len(sentences) * self.sample_size))
        selected_sentences = random.sample(sentences, sample_size)
        result = ' '.join(selected_sentences)
        return result

    def get_content_from_file(self):

        file_content = ''

        if self.localPath.endswith('.csv'):
            df = pd.read_csv(self.localPath)
            df = df.astype(str)
            if self.randomize:
                df = self.get_random_csv_chunk(df)
            file_content = df.to_string(header=True, index=False, max_rows=None)
        elif self.localPath.endswith('.txt'):
            with open(self.localPath, 'r') as file:
                file_content = file.read()
                if self.randomize:
                    file_content = self.get_random_text_chunk(file_content)
        elif self.localPath.endswith('.pdf'):
            reader = PdfReader(self.localPath)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            if self.randomize:
                file_content = self.get_random_text_chunk(text)
            else:
                file_content = text
        else:
            raise ValueError("Unsupported file format")

        self.content = file_content
        return file_content
    


    def get_content_from_directory(self):
        """
        Iterates over files in the directory, reads their content,
        and concatenates it into a single string.
        """
        concatenated_content = ''
        for filename in os.listdir(self.directoryPath):
            filepath = os.path.join(self.directoryPath, filename)
            if filepath.endswith(('.csv', '.txt', '.pdf')):
                self.localPath = filepath  # Temporarily update the localPath
                file_content = self.get_content_from_file()
                concatenated_content += file_content + '\n'

        self.content = concatenated_content  # Store concatenated content
        return concatenated_content


class RAGInternet(BaseModel):

    keywords:List[str]
    return_chunks: Optional[bool] = False
    minimum_number_of_words_by_article: Optional[int] = 500
    maximum_number_of_words_by_article: Optional[int] = 50000
    content: Optional[str] = None

    def word_counter(self, input_string):
        # Split the string into words based on whitespace
        words = input_string.split()

        # Count the number of words
        number_of_words = len(words)

        return number_of_words

    def get_google_search_result(self, keyword:dict, maximum_number_of_link:int = None):

        encoded_keyword = quote_plus(keyword)

        url = f"https://api.serply.io/v1/search/q={encoded_keyword}"

        headers = {
            "Content-Type": "application/json",
            "X-User-Agent": "",
            "X-Proxy-Location": "",
            "X-Api-Key": os.environ.get("SERPLY_API_KEY"),
            "X-Proxy-Location": "US"
        }

        response = requests.request("GET", url, headers=headers)

        response_json = json.loads(response.text)["results"]

        result = []

        for element in response_json:

            link = element['link']
            result.append(link)

        if maximum_number_of_link:
            return result[:maximum_number_of_link]

        return result

    def get_content_from_url(self, link:str):

        downloaded = trafilatura.fetch_url(link)
        content = trafilatura.extract(downloaded)

        return content

    def extract_content_from_internet(self):

        print(f"Browsing...")

        for keyword in self.keywords:

            result = ""

            urls = self.get_google_search_result(keyword)

            for url in urls:

                content = self.get_content_from_url(url)

                if content and self.word_counter(content) > self.minimum_number_of_words_by_article and self.word_counter(content) < self.maximum_number_of_words_by_article:

                    print(url)

                    result = result + "\n" + content

        print("Finish browsing...")
        self.content = result
        return result

class Validator(BaseModel):

    function_name:str
    additional_parameters:Optional[List[str]] = None
    from_notebook:bool = False
    retry_number:Optional[int] = 3


class Variations(BaseModel):

    id:str
    parent_id:Optional[str] = None
    value:str
    confidence_score:Optional[float] = None  
    error_message:str = None
    model_used:str = None

    class Config:
        extra = "forbid"  # This will raise an error for extra fields
 

class Decontomination(BaseModel):

    embedding_model:EmbeddingModel 
    threshold: Optional[float] = 0.99
    column_name: Optional[str] = None 
    delete_column: Optional[str] = None 
    delete_mode: Optional[DeleteMode] = DeleteMode.HIGHEST

    def decontaminate_variable(self, variations:Dict[str, Variations]):
        
        model = self.embedding_model.get_model()

        data: list[Variations] = list(variations.values())

        embeddings = [model.create_embedding(row.value) for row in data]
        embeddings_np = np.array(embeddings)
    
        sim_matrix = np.zeros((len(embeddings_np), len(embeddings_np)))

        for i in range(len(embeddings_np)):
            for j in range(len(embeddings_np)):
                sim_matrix[i][j] = cosine_similarity(embeddings_np[i], embeddings_np[j])
        
        exclude_indices = set()

        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if sim_matrix[i][j] > self.threshold:
                    exclude_indices.add(j)

        decontamined_variations = {key: variations[key] for i, key in enumerate(variations) if i not in exclude_indices}

        return decontamined_variations



    def decontaminate(self, data: List[Dict]):

        model = self.embedding_model.get_model()
        
        embeddings = [model.create_embedding(row[self.column_name]) for row in data]
        embeddings_np = np.array(embeddings)

        # Calculate cosine similarity matrix manually
        sim_matrix = np.zeros((len(embeddings_np), len(embeddings_np)))

        for i in range(len(embeddings_np)):
            for j in range(len(embeddings_np)):
                sim_matrix[i][j] = cosine_similarity(embeddings_np[i], embeddings_np[j])
        
        # Identify rows to keep (those that don't have too high similarity with any other row)
        rows_to_keep = []
        # Mark rows for exclusion based on similarity threshold
        exclude_indices = set()

        for i in range(len(data)):
            for j in range(i + 1, len(data)):  # Compare each pair once
                if sim_matrix[i][j] > self.threshold:
                    # Mark the row with a lower value in delete_column for exclusion
                    if self.delete_column:
                        if self.delete_mode == DeleteMode.HIGHEST and data[i][self.delete_column] < data[j][self.delete_column]:
                            exclude_indices.add(i)
                        elif self.delete_mode == DeleteMode.LOWEST and data[i][self.delete_column] > data[j][self.delete_column]:
                            exclude_indices.add(i)
                        else:
                            exclude_indices.add(j)
                    else:
                        # If no delete_column is specified, exclude one of the rows arbitrarily
                        exclude_indices.add(j)

        # Identify rows to keep (those not marked for exclusion)
        rows_to_keep = [row for idx, row in enumerate(data) if idx not in exclude_indices]

        return rows_to_keep


class Junction(BaseModel):

    value:Optional[str] = None 
    model:Optional[Model] = None 
    delete_branch:Optional[bool] = False 

    class Config:
        extra = "forbid"
    
    def generate(self, data:List[str]):

        current_model = self.model.get_model()

        prompt = "Given this following values:"
    
        for val in data:
            prompt += f"\n'''\n{val}\n'''\n"

        if isinstance(current_model, OpenAIInstructModel) or isinstance(current_model, LlamaCPPModel):

            start_messages = prompt
        
        elif isinstance(current_model, OpenAIChatModel):

            start_messages = [
                {"role": "system", "content": current_model.system_prompt},
                {"role": "user", "content": prompt},
            ]   
            
        elif isinstance(current_model, MistralChatModel):

            start_messages = [
                ChatMessage(role="system", content= current_model.system_prompt),
                ChatMessage(role="user", content=prompt)
            ]

        elif isinstance(current_model, TogetherChatModel):
        
            start_messages = [
                {"role": "system", "content": current_model.system_prompt},
                {"role": "user", "content": prompt},
            ] 

        elif isinstance(current_model, AnyscaleChatModel):
        
            start_messages = [
                {"role": "system", "content": current_model.system_prompt},
                {"role": "user", "content": prompt},
            ]
            
        else:

            raise ValueError("Unknow type of model")

        generated_value = current_model.ask(messages=start_messages)

        self.value = generated_value

        return generated_value


class Variable(BaseModel):

    name: str
    models:Optional[List[Model]] = None
    ensure_model_diversity:Optional[bool] = False 
    generation_number: int = 1
    source_internet: Optional[RAGInternet] = None
    source_localfile: Optional[RAGLocalPath] = None
    source_localdirectory: Optional[RAGLocalPath] = None
    source_huggingface:Optional[RAGHuggingFace] = None
    get_value_from_huggingface:Optional[RAGHuggingFace] = None
    get_value_from_localfile:Optional[RAGLocalPath] = None
    get_value_from_custom_functions:Optional[Validator] = None
    transform_value:Optional[Validator] = None 
    note: Optional[List[str]] = None
    rag_content: Optional[str] = None
    validator:Optional[Validator] = None
    decontamination:Optional[Decontomination] = None 
    values:Optional[Dict[str, Variations]] = {}
    junction:Optional[Junction] = None 

    model_config = ConfigDict(
            protected_namespaces=('protect_me_', 'also_protect_'),
            extra = "forbid"
        )

    def load_internet_source(self):

        if self.source_internet is not None:
            self.rag_content = self.source_internet.extract_content_from_internet()

    def load_local_file(self):

        if self.source_localfile is not None and self.source_localfile.localPath is not None:
            self.rag_content = self.source_localfile.get_content_from_file()

    def load_local_directory(self):

        if self.source_localfile is not None and self.source_localfile.directoryPath is not None:
            self.rag_content = self.source_localfile.get_content_from_directory()

    def load_huggingface_dataset(self):

        if self.source_huggingface is not None:
            self.rag_content = self.source_huggingface.get_random_value_from_dataset()

    def load_value(self):
        
        if self.get_value_from_huggingface:
            self.value = self.get_value_from_huggingface.get_random_value_from_dataset(max_token=self.max_tokens)





class Template(BaseModel):

    description: str
    prompt: str
    completion: str
    prompt_variation_number: Optional[int] = 1
    variables: Optional[Dict[str, Variable]] = None
    source_internet: Optional[RAGInternet] = None
    source_localfile: Optional[RAGLocalPath] = None
    rag_content: Optional[str] = None
    value:Optional[List[str]] = None
    decontamination: Optional[Decontomination] = None 
    
    class Config:
        extra = "forbid"  

    def load_internet_source(self):

        if self.source_internet is not None:
            self.rag_content = self.source_internet.extract_content_from_internet()

    def load_local_file(self):

        if self.source_localfile is not None and self.source_localfile.localPath is not None:
            self.rag_content = self.source_localfile.get_content_from_file()

    def load_local_directory(self):

        if self.source_localfile is not None and self.source_localfile.directoryPath is not None:
            self.rag_content = self.source_localfile.get_content_from_directory()

class TemplateName(Enum):
    PRODUCT_REVIEW = "product-review"
    CHUNK = "chunk"
    CHUNK2 = "chunk2"
    HALLUCINATION = "hallucination"


class TemplateManager:

    def __init__(self, template_file_path:str):
        self.template_file_path = self.get_template_file_path(template_file_path)
        self.templates = self.load_templates()

    def get_template_file_path(self, filename: str) -> str:
        base_path = os.getcwd()  
        
        if os.path.isabs(filename):
            return filename
        else:
            return os.path.join(base_path, filename) 

    def load_templates(self) -> Dict[str, Template]:
        with open(self.template_file_path, 'r') as file:
            raw_data = json.load(file)

        templates = {}
        for key, data in raw_data.items():
            
            template_name = key
            template = Template(**data)
            templates[template_name] = template
            
        return templates

    def get_template(self, template_name: str) -> Template:

        template = self.templates.get(template_name)

        if template:
            
            template.load_internet_source()
            template.load_local_file()
            template.load_local_directory()

        return template

def create_variable_from_name(model:OpenAIChatModel, variable_name:str) -> Variable:

    prompt = load_file(path="files/variable_generation.txt")

    prompt = prompt.format(variable_name=variable_name)

    completion = model.ask_instruct_gpt(prompt=prompt, temperature=0, max_tokens=30)

    return Variable(**completion)
