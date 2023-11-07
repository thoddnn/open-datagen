import os 
import csv 
import trafilatura
import re 

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