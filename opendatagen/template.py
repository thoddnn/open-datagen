from pydantic import BaseModel, validator, ValidationError, ConfigDict
from typing import Optional, List, Dict, Union
from enum import Enum
import os
import json
from opendatagen.utils import load_file
from opendatagen.model import OpenAIChatModel, OpenAIInstructModel, OpenAIEmbeddingModel, HuggingFaceModel, Model
from urllib.parse import quote_plus
import requests
import trafilatura
from PyPDF2 import PdfReader
import pandas as pd
from datasets import load_dataset, Dataset
from opendatagen.utils import get_first_n_tokens, num_tokens_from_string
import random
import uuid


class RAGHuggingFace(BaseModel):

    dataset_path:str
    dataset_name:Optional[str] = None
    data_dir:Optional[str] = None
    column_name:str
    streaming:bool = True
    min_tokens:Optional[int] = 0
    max_tokens:Optional[int] = None
    subset_size:Optional[int] = 10000

    class Config:
        extra = "forbid"

    def get_random_value_from_dataset(self):

        param = {}

        if self.dataset_path:
            param["path"] = self.dataset_path

        if self.data_dir:
            param["data_dir"] = self.data_dir

        if self.dataset_name:
            param["name"] = self.dataset_name

        param["streaming"] = self.streaming

        dst = load_dataset(**param)

        subset = [sample[self.column_name] for _, sample in zip(range(self.subset_size), dst["train"])]

        max_attempts = 100
        count = 0

        while count < max_attempts:

            index = random.randint(0, len(subset) - 1)

            text = subset[index]

            num_tokens = num_tokens_from_string(text, encoding_name="cl100k_base")

            if num_tokens >= self.min_tokens:

                if self.max_tokens:

                    text = subset[index]

                    result = get_first_n_tokens(n=self.max_tokens, text=text, encoding_name="cl100k_base")
                    
                    return result

                else:

                    result = subset[index]

                    return result

            count = count + 1



    """
    def get_n_nearest_value_from_dataset(self, n:int=1, prompt:str)
    """


class RAGLocalPath(BaseModel):

    localPath:Optional[str] = None
    directoryPath:Optional[str] = None
    content:Optional[str] = None

    class Config:
        extra = "forbid"

    def get_content_from_file(self):
        """
        Reads the content from a file based on its extension.
        Handles CSV, TXT, and PDF files.
        """
        file_content = ''
        if self.localPath.endswith('.csv'):
            df = pd.read_csv(self.localPath)
            df = df.astype(str)
            file_content = df.to_string(header=True, index=False, max_rows=None)
            print(file_content)
        elif self.localPath.endswith('.txt'):
            with open(self.localPath, 'r') as file:
                file_content = file.read()
        elif self.localPath.endswith('.pdf'):
            reader = PdfReader(self.localPath)
            for page in reader.pages:
                file_content += page.extract_text() + '\n'
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
    error_message:str = None

    class Config:
        extra = "forbid"  # This will raise an error for extra fields
 


class Variable(BaseModel):

    name: str
    models:Optional[List[Model]] = None 
    generation_number: int = 1
    source_internet: Optional[RAGInternet] = None
    source_localfile: Optional[RAGLocalPath] = None
    source_localdirectory: Optional[RAGLocalPath] = None
    source_huggingface:Optional[RAGHuggingFace] = None
    get_value_from_huggingface:Optional[RAGHuggingFace] = None
    get_value_from_localfile:Optional[RAGLocalPath] = None
    note: Optional[List[str]] = None
    rag_content: Optional[str] = None
    validator:Optional[Validator] = None
    values:Optional[Dict[str, Variations]] = {}

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

    class Config:
        extra = "forbid"  # This will raise an error for extra fields

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
        #self.template_file_path = self.get_template_file_path(filename)
        self.templates = self.load_templates()

    def get_template_file_path(self, filename: str) -> str:
        return os.path.join(os.path.dirname(__file__), filename)

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
