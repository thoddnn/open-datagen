import os
import openai
from dotenv import load_dotenv
from enum import Enum
import numpy as np
import trafilatura
import time
import random
import re
import json
import requests
from urllib.parse import quote
import csv
from tenacity import retry, stop_after_attempt, wait_exponential
from re import findall
from pydantic import BaseModel, validator, conint, constr, ValidationError
from typing import Optional, Dict, List 

load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

openai.api_key = openai_api_key

serply_api_key = os.environ.get('SERPLY_API_KEY')
if not serply_api_key:
    raise ValueError("SERPLY_API_KEY not found in environment variables.")

N_RETRIES = 3

output_array = []

class Models(Enum): 
    GPT_35_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    TEXT_DAVINCI_INSTRUCT= "text-davinci-003"
    GPT_35_TURBO_CHAT = "gpt-3.5-turbo"
    GPT_35_TURBO_16K_CHAT = "gpt-3.5-turbo-16k"
    GPT_4_CHAT = "gpt-4"

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
def ask_chat_gpt(model:str, system_prompt:str, user_prompt:str, max_tokens:int, temperature:int) -> str: 

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return completion.choices[0].message["content"]

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=60))
def ask_instruct_gpt(model:str, prompt:str, temperature:int, max_tokens:int) -> str:

    completion = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return completion.choices[0].text
    
def create_embedding(prompt:str):

    embedding = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=prompt
    )

    return embedding["data"][0]["embedding"]


def cosine_similarity(A:list, B:list):

    return np.dot(A, B)


def extract_website_details(url):
    downloaded = trafilatura.fetch_url(url)
    metadata = trafilatura.metadata.extract_metadata(downloaded)

    title = metadata['title'] if metadata and 'title' in metadata else None
    description = metadata['description'] if metadata and 'description' in metadata else None
    
    content = trafilatura.extract(downloaded)

    response = {
        "title": title,
        "description": description,
        "content": content
    }

    return response

def get_k_nearest(user_prompt, k, embeddings_dict):
    # Compute the embedding for the user's prompt
    user_embedding = create_embedding(prompt=user_prompt)
    
    # Calculate cosine similarity scores with all stored embeddings
    similarities = {}

    for sub_topic, emb in embeddings_dict.items():
        sim = cosine_similarity(user_embedding, emb) 
        similarities[sub_topic] = sim

    # Sort sub-topics by similarity and retrieve the top K
    sorted_subtopics = sorted(similarities, key=similarities.get, reverse=True)[:k]
    
    return sorted_subtopics

def load_file(path:str):
    # Adjust the path based on this module's location
    absolute_path = os.path.join(os.path.dirname(__file__), path)
    
    with open(absolute_path, 'r') as file:
        content = file.read()
    
    return content


def chunk_text(text, x):
    """
    Chunks a given text into pieces of approximately x words, ensuring complete phrases.

    :param text: The input text.
    :param x: Desired number of words per chunk.
    :return: List of text chunks.
    """
    # Split text into phrases by punctuation marks that typically end sentences or clauses.
    phrases = re.split(r'(?<=[.!?;])\s+', text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for phrase in phrases:
        phrase_word_count = len(phrase.split())

        # If adding the next phrase won't exceed x words, add it to the current chunk.
        if current_word_count + phrase_word_count <= x:
            current_chunk.append(phrase)
            current_word_count += phrase_word_count
        else:
            # If the current chunk isn't empty, add it to the list of chunks.
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # If a single phrase has more than x words, we'll have to split it.
            if phrase_word_count > x:
                words = phrase.split()
                while words:
                    space_left = x - current_word_count
                    current_chunk.extend(words[:space_left])
                    words = words[space_left:]
                    current_word_count = len(current_chunk)

                    if current_word_count == x or not words:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_word_count = 0
            else:
                # Otherwise, just start a new chunk with the current phrase.
                current_chunk.append(phrase)
                current_word_count = phrase_word_count

    # Add any remaining chunk to the list.
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def google_search(query:str, location:str):

    encoded_query = quote(query)

    url = "https://api.serply.io/v1/search/q=" + encoded_query

    headers = {
        "Content-Type": "application/json",
        "X-User-Agent": "",
        "X-Proxy-Location": location,
        "X-Api-Key": serply_api_key
    }

    response = requests.request("GET", url, headers=headers)

    return json.loads(response.text)

def google_news_search(query:str, location:str):

    encoded_query = quote(query)

    url = "https://api.serply.io/v1/news/q=" + encoded_query

    headers = {
        "Content-Type": "application/json",
        "X-User-Agent": "",
        "X-Proxy-Location": location,
        "X-Api-Key": serply_api_key
    }

    response = requests.request("GET", url, headers=headers)

    return json.loads(response.text)


class Variable(BaseModel):
    name: str
    temperature: float
    max_tokens: int
    generation_number: int
    type: Optional[str]  # like 'int' in your example
    min_value: Optional[int]  # constrain integer to be >= 0
    max_value: Optional[int]
    start_with: Optional[List[str]]
    note: Optional[str]
    rag_content: Optional[str]

    # Validate the type value if it's provided
    @validator('type', pre=True, always=True)
    def check_type(cls, value):
        if value and value not in ['int', 'str']:
            raise ValueError(f'Invalid type: {value}')
        return value
    
    class Config:
        extra = "forbid"  # This will raise an error for extra fields

class Template(BaseModel):
    description: str
    prompt: str
    completion: str
    prompt_variables: Optional[Dict[str, Variable]]
    completion_variables: Optional[Dict[str, Variable]]
    prompt_variation_number: Optional[int] = 5  

    class Config:
        extra = "forbid"  # This will raise an error for extra fields

class TemplateName(Enum):
    PRODUCT_REVIEW = "product-review"
    BLOG_POST = "blog-post"
    # Add other templates as needed


class TemplateManager:

    def __init__(self, template_file=None):
        if not template_file:
            # Default to the bundled template file based on this module's location
            template_file = os.path.join(os.path.dirname(__file__), 'files', 'template.json')
        
        with open(template_file, 'r') as file:
            self.templates = json.load(file)
        
    def list_templates(self):
        """List all available templates with descriptions."""
        for name, details in self.templates.items():
            print(f"{name}: {details['description']}")
            
    def get_template(self, template_name="default"):
        """Retrieve a specific template by name."""
        return self.templates.get(template_name, None)
        
    def use_template(self, template_name, prompt_data, completion_data):
        """Use a specific template and populate it with provided data."""
        template = self.templates.get(template_name, None)
        if not template:
            raise ValueError(f"Template {template_name} not found!")
        
        prompt = template['prompt'].format(**prompt_data)
        completion = template['completion'].format(**completion_data)
        
        return prompt, completion

    

def dict_to_string(d):
    result = []
    for key, value in d.items():
        result.append(f'#{key}#:\n"""')
        result.append(f'{value}')
        result.append('"""')
    return '\n'.join(result)


def process_variations(index, variables_list, variables_dict, template):
    # Base case: If index is out of range of variables_list, return
    if index > len(variables_list) - 1:

        output_array.append(variables_dict.copy()) 

        return 

    # Extract variable names
    variable_name = variables_list[index]

    # Outer loop
    variations = generate_variation(variable_name=variable_name, variables_dict=variables_dict, template=template)["variations"]

    for variation_value in variations:
        # Update dictionary
        variables_dict[variable_name] = variation_value

        # Reset subsequent dictionary keys to => need to have ordered keys => python 3.8""
        for i in range(index + 1, len(variables_list)):
            variables_dict[variables_list[i]] = ""

        process_variations(index + 1, variables_list, variables_dict, template=template)

def generate_context_from_json(data, stop_field=None):
    if stop_field and list(data.keys())[0] == stop_field:
        return ""

    output = "Given these values\n"
    
    for key, value in data.items():
        if key == stop_field:
            break
        output += f"#{key} value#\n'''{value}\n'''\n"
    
    return output


def generate_variation(variable_name:str, variables_dict:dict, template:Template):
    
    initial_variation_prompt = load_file(path="files/generation.txt")

    temp_variation_prompt = initial_variation_prompt

    # Sample JSON
    context = generate_context_from_json(variables_dict, variable_name)

    max_tokens = int(template.prompt_variables[variable_name].max_tokens)
    temperature = template.prompt_variables[variable_name].temperature
    number_of_generation = template.prompt_variables[variable_name].generation_number
    name = template.prompt_variables[variable_name].name
    
    note = template.prompt_variables[variable_name].note

    start_with = template.prompt_variables[variable_name].start_with or ""

    var_type = template.prompt_variables[variable_name].type or ""

    rag_content = template.prompt_variables[variable_name].rag_content or ""

    if rag_content != "":
        rag_content = "Here are some examples that might help you:\n\n" + rag_content
    
    type_constraint = ""
    
    if var_type == "int":
        type_constraint = "The variations must be integer."

    last_values_list = []
    last_values = ""

    for _ in range(number_of_generation):
      
        temp_variation_prompt = initial_variation_prompt.format(json=variables_dict, 
                                                         variable_name=name,
                                                         rag_content=rag_content, 
                                                         start_with=start_with,
                                                         last_values=last_values,
                                                         type_constraint = type_constraint,
                                                         note=note,
                                                         context=context)
        
        variation_completion = ask_chat_gpt(model=Models.GPT_35_TURBO_CHAT.value, system_prompt="No verbose.",
                                        user_prompt=temp_variation_prompt, max_tokens=max_tokens, temperature=temperature)
        
        last_values_list.append(variation_completion)

        # Create the desired string format if last_values_list is not empty
        if last_values_list:
            last_values = "Generate a content value that is not similar to following values:\n" + "\n".join(last_values_list)
        else:
            last_values = ""

    #variation_completion JSON string sometimes contains ' instead of "
    variation_completion = variation_completion.replace("'", '"')

    variations = {"variations": last_values_list}

    return variations

def create_type_message(comp_type, min_value, max_value):
    """Helper function to create the type message based on the given constraints."""
    type_msg = f"The answer must be a {comp_type}" if comp_type else ""

    if comp_type == "int":
        if min_value and max_value:
            type_msg += f" between {min_value} and {max_value}"
        elif max_value:
            type_msg += f" lower than {max_value}"
        elif min_value:
            type_msg += f" greater than {min_value}"

    return type_msg


def write_to_csv(rows, path):
    """Write the provided data to a CSV file at the specified path."""
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()  # Writing the headers
        writer.writerows(rows)


# Initialize the template manager and retrieve the template
def generate_data(template:Template, output_path:str):

    result = []

    # Extracting structures and variables from the template
    prompt = template.prompt 
    completion = template.completion

    prompt_variables = findall(r'\{(.*?)\}', prompt)
    completion_variables = findall(r'\{(.*?)\}', completion)

    prompt_var_dict = {var: "" for var in prompt_variables}

    process_variations(index=0, 
                        variables_list=prompt_variables, 
                        variables_dict=prompt_var_dict, 
                        template=template)

    # Loading files outside loops to reduce overhead
    evol_prompt_template = load_file(path="files/evol_instruct.txt")
    completion_prompt_template = load_file(path="files/completion.txt")

    # Always ask to generate X variations for the same prompt
    evol_number = template.prompt_variation_number
    
    for param in output_array:
        
        generated_prompt = prompt.format(**param)
        
        # Formatting the reference prompt
        reference_prompt = evol_prompt_template.format(number_of_prompts=str(evol_number), prompt=generated_prompt)
        
        # Generate diversified prompts 
        prompt_diversified_response = ask_chat_gpt(model=Models.GPT_35_TURBO_CHAT.value, 
                                                    system_prompt="Answer as a valid JSON like {\"prompts\": [\"XXXX\", \"YYYY\"]}",
                                                    user_prompt=reference_prompt, 
                                                    max_tokens=512, 
                                                    temperature=1)
        
        diversified_prompts = json.loads(prompt_diversified_response)["prompts"]
        
        for diversified_prompt in diversified_prompts[:evol_number]:

            completion_results = {}

            initial_diversified_prompt = diversified_prompt

            for placeholder in completion_variables:
                completion_config = template.completion_variables[placeholder]

                completion_temperature = completion_config.temperature
                max_tokens = completion_config.max_tokens
                comp_type = completion_config.type
                min_value = completion_config.min_value
                max_value = completion_config.max_value
                start_options = completion_config.start_with

                type_message = create_type_message(comp_type, min_value, max_value)

                start_with = random.choice(start_options) if start_options and start_options != [""] else ""

                completion_prompt = completion_prompt_template.format(prompt=diversified_prompt,
                                                                        variable_name=completion_config.name,
                                                                        completion_type=type_message,
                                                                        start_with=start_with)

                completion_content = ask_instruct_gpt(model=Models.GPT_35_TURBO_INSTRUCT.value,
                                                        prompt=completion_prompt,
                                                        temperature=completion_temperature,
                                                        max_tokens=max_tokens)
                
                completion_results[placeholder] = completion_content

                diversified_prompt = f"{diversified_prompt}\n\n'''{completion_content}'''"
            
            final_output = completion.format(**completion_results)

            # Append the generated data to the result list
            row = {"prompt": generated_prompt, "diversified_prompt": initial_diversified_prompt, "completion": final_output}
            row.update(param)
            row.update(completion_results)
            result.append(row)

            # Save the partial result as CSV
            write_to_csv(result, output_path)

    return result


if __name__ == "__main__":

    #manager = TemplateManager()
    #template = manager.get_template(template_name=TemplateName.PRODUCT_REVIEW.value)
    #generate_data(template=template, output_path="output.csv")

    # Create the custom template using the Pydantic models
    user_template = Template(
        description="Custom template for Python exercises",
        prompt="Python exercise: '{python_exercise}'",
        completion="Answer using python:\n---\n{python_code}\n---",
        prompt_variation_number=1,
        prompt_variables={
            "python_exercise": Variable(
                name="Python exercice",
                temperature=1,
                max_tokens=126,
                generation_number=5,
                note="The python exercise statement must be medium level."
            
            )
        },
        completion_variables={
            "python_code": Variable(
                name="Python code",
                temperature=0,
                max_tokens=256,
                generation_number=1
            )
        }
    )

    data = generate_data(template=user_template, output_path="output.csv")

    '''
    user_template = {}

    # adding necessary details to the custom template
    # the prompt and completion structure
    user_template["prompt"] = "Python exercise: '{python_exercise}'"
    user_template["completion"] = "Answer using python:\n---\n{python_code}\n---"

    # defining prompt variables details - name, temperature, max_tokens, and number of variations.
    user_template["prompt_variables"] = {
        "python_exercise": {"name": "Python exercice", "temperature":1, "max_tokens":50, "variation_nb":1, "note": "The python exercice statement must be easy."}
    }

    # defining completion variables details - name, temperature, max_tokens, and their data type.
    # 'start_with' is an optional parameter to specify the beginning of the completion.
    user_template["completion_variables"] = {
        "python_code": {"name": "Python code", "temperature":0, "max_tokens":512, "variation_nb":1}
    }

    generate_data(template=user_template, output_path="output.csv")
    '''