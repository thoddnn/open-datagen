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


class TemplateName(Enum):
    DEFAULT = "default"
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
        
        prompt = template['prompt_structure'].format(**prompt_data)
        completion = template['completion_structure'].format(**completion_data)
        
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


def generate_variation(variable_name:str, variables_dict:dict, template:dict):
    
    temp_variation_prompt = load_file(path="files/variations.txt")

    max_tokens = int(template["prompt_variables"][variable_name]["max_tokens"])
    temperature = template["prompt_variables"][variable_name]["temperature"]
    number_of_variation = int(template["prompt_variables"][variable_name]["variation_nb"])
    name = str(template["prompt_variables"][variable_name]["name"])
    
    note = template["prompt_variables"][variable_name].get("note", "") 

    start_with = template["prompt_variables"][variable_name].get("start_with", "") 

    var_type = template["prompt_variables"][variable_name].get("type", "") 

    rag_content = template["prompt_variables"][variable_name].get("rag_content", "")

    if rag_content != "":
        rag_content = "Here are some examples that might help you:\n\n" + rag_content

    temp_variation_prompt = temp_variation_prompt.format(json=variables_dict, 
                                                         number_of_variation=str(number_of_variation), 
                                                         variable=variable_name,
                                                         rag_content=rag_content, 
                                                         start_with=start_with,
                                                         note=note)

    if var_type == "int":
        
        temp_variation_prompt = temp_variation_prompt + "\n" + "The variations must be integer."


    variation_completion = ask_chat_gpt(model=Models.GPT_35_TURBO_CHAT.value, system_prompt="Answer as a valid JSON like \{'variations': ['XXXX', 'YYYY']\}",
                                        user_prompt=temp_variation_prompt, max_tokens=max_tokens, temperature=temperature)

    variations = json.loads(variation_completion)

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
        writer = csv.writer(file)
        writer.writerow(["initial_prompt", "evolution_prompt", "completion"])  # Writing the headers
        writer.writerows(rows)




# Initialize the template manager and retrieve the template

def generate_data(template, output_path):

    csv_rows = []

    # Extracting structures and variables from the template
    prompt_structure = template["prompt_structure"]
    completion_structure = template["completion_structure"]

    prompt_variables = findall(r'\{(.*?)\}', prompt_structure)
    completion_placeholders = findall(r'\{(.*?)\}', completion_structure)

    prompt_var_dict = {var: "" for var in prompt_variables}

    variations = generate_variation(variable_name=prompt_variables[0], 
                                    variables_dict=prompt_var_dict, 
                                    template=template)["variations"]

    process_variations(index=0, 
                        variables_list=prompt_variables, 
                        variables_dict=prompt_var_dict, 
                        template=template)

    # Loading files outside loops to reduce overhead
    evol_prompt_template = load_file(path="files/evol_instruct.txt")
    completion_prompt_template = load_file(path="files/completion.txt")

    for param in output_array:
        
        generated_prompt = prompt_structure.format(**param)
        
        # Formatting the reference prompt
        reference_prompt = evol_prompt_template.format(number_of_prompts=str(1), prompt=generated_prompt)
        
        prompt_diversified_response = ask_chat_gpt(model=Models.GPT_35_TURBO_CHAT.value, 
                                                    system_prompt="Answer as a valid JSON like {'prompts': ['XXXX', 'YYYY']}",
                                                    user_prompt=reference_prompt, 
                                                    max_tokens=256, 
                                                    temperature=1)

        diversified_prompts = json.loads(prompt_diversified_response)["prompts"]

        for diversified_prompt in diversified_prompts:

            initial_diversified_prompt = diversified_prompt  # Store the initial value of diversified_prompt

            completion_results = {}

            for placeholder in completion_placeholders:
                completion_config = template["completion_variables"][placeholder]

                completion_temp = completion_config.get("temperature", "")
                max_tokens = completion_config.get("max_tokens", "")
                comp_type = completion_config.get("type", "")
                min_value = completion_config.get("min_value", "")
                max_value = completion_config.get("max_value", "")
                start_options = completion_config.get("start_with", [""])

                type_message = create_type_message(comp_type, min_value, max_value)

                start_with = random.choice(start_options) if start_options and start_options != [""] else ""

                completion_prompt = completion_prompt_template.format(prompt=diversified_prompt,
                                                                        variable_name=completion_config["name"],
                                                                        completion_type=type_message,
                                                                        start_with=start_with)

                completion_content = ask_instruct_gpt(model=Models.GPT_35_TURBO_INSTRUCT.value,
                                                        prompt=completion_prompt,
                                                        temperature=completion_temp,
                                                        max_tokens=max_tokens)
                
                completion_results[placeholder] = completion_content

                diversified_prompt = f"{diversified_prompt}\n{completion_content}"
            
            final_output = completion_structure.format(**completion_results)

            # Append the generated data to the csv_rows list
            csv_rows.append([generated_prompt, initial_diversified_prompt, final_output])
            

    write_to_csv(csv_rows, output_path)


