import os
import csv
import trafilatura
import re
import requests
from urllib.parse import quote_plus
import json
import importlib
import tiktoken
from datasets import Dataset
import random

def dict_to_string(d):
    result = []
    for key, value in d.items():
        result.append(f'#{key}#:\n"""')
        result.append(f'{value}')
        result.append('"""')
    return '\n'.join(result)

def load_file(path:str):
    # Adjust the path based on this module's location
    absolute_path = os.path.join(os.path.dirname(__file__), path)

    with open(absolute_path, 'r') as file:
        content = file.read()

    return content

def write_to_csv(rows, path):
    """Write the provided data to a CSV file at the specified path."""
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()  # Writing the headers
        writer.writerows(rows)

def generate_context_from_json(data, stop_field=None):
    if stop_field and list(data.keys())[0] == stop_field:
        return ""

    output = "Given these values\n"

    for key, value in data.items():
        if key == stop_field:
            break
        output += f"#{key} value#\n'''{value}\n'''\n"

    return output


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

def find_strings_in_brackets(text):
    # This pattern matches text enclosed in { and }
    pattern = r"\{(.*?)\}"
    # Find all matches
    matches = re.findall(pattern, text)
    return matches

def snake_case_to_title_case(snake_str):
    # Split the string at underscores
    words = snake_str.split('_')
    # Capitalize the first letter of each word and join them with a space
    title_case_str = ' '.join(word.capitalize() for word in words)
    return title_case_str

def title_case_to_snake_case(title_str):
    # First, split the string by spaces
    words = title_str.split(' ')
    # Convert all the words to lowercase and join them with underscores
    snake_case_str = '_'.join(word.lower() for word in words)
    return snake_case_str



def word_counter(input_string):
    # Split the string into words based on whitespace
    words = input_string.split()

    # Count the number of words
    number_of_words = len(words)

    return number_of_words

def get_google_search_result(keyword:dict, maximum_number_of_link:int = None):

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

def get_content_from_url(link:str):

    downloaded = trafilatura.fetch_url(link)
    content = trafilatura.extract(downloaded)

    return content

def extract_content_from_internet(keyword:str):

    print(f"Browsing for the keyword {keyword}...")

    result = ""

    urls = get_google_search_result(keyword)

    for url in urls:

        content = get_content_from_url(url)

        if content and word_counter(content) > 500:

            print(url)

            result = result + "\n" + content

    print("Finish browsing...")

    return result


def load_user_function(full_function_name:str, from_notebook:bool):
    if from_notebook:
        try:
            from IPython import get_ipython
            ipython_namespace = get_ipython().user_ns
        except ImportError:
            raise EnvironmentError("IPython environment not detected for notebook mode.")

        if full_function_name in ipython_namespace:
            func = ipython_namespace[full_function_name]
            if callable(func):
                return func
            else:
                raise TypeError(f"The object '{full_function_name}' in the IPython namespace is not callable.")
        else:
            raise ValueError(f"Function '{full_function_name}' not found in the IPython namespace.")
    else:
        try:
            module_name, function_name = full_function_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)
        except ValueError:
            raise ValueError(f"Invalid format for function name '{full_function_name}'. Expected 'module.function_name'.")
        except ImportError:
            raise ImportError(f"Module '{module_name}' could not be found.")
        except AttributeError:
            raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'.")

        if not callable(func):
            raise TypeError(f"The object '{function_name}' found in module '{module_name}' is not callable.")

        return func


def function_to_call(function_name, from_notebook, *args):

    user_function = load_user_function(function_name, from_notebook)

    return user_function(*args)

def is_retryable_answer(result):
    if "i can't fulfill that request" in result.lower():
        return True
    else:
        return False


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_first_n_tokens(text: str, encoding_name: str, n: int, cut_last_sentence: bool = False) -> str:
    """Returns the first n tokens of a string, with an option to cut the last sentence."""
    # Encode the string into tokens
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    # Retrieve the first n tokens
    tokens = tokens[:n]

    # Cut the last sentence if required
    if cut_last_sentence:
        for i in range(len(tokens) - 1, -1, -1):
            # Assuming '.' represents the end of a sentence
            if encoding.decode([tokens[i]]) == '.':
                tokens = tokens[:i+1]
                break

    # Decode the tokens back to string
    return encoding.decode(tokens)
