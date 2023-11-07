
from dotenv import load_dotenv
import numpy as np
import time
import random
import re
import json
import requests
from urllib.parse import quote
from re import findall
from utils import dict_to_string, load_file, write_to_csv, generate_context_from_json, extract_website_details, create_type_message, find_strings_in_brackets
from utils import snake_case_to_title_case, title_case_to_snake_case
from anonymizer import Anonymizer
from model import OpenAIModel, ModelName
from template import Template, Variable,  create_variable_from_name


load_dotenv()

class DataGenerator:

    output_array = []

    def __init__(self, variation_model:OpenAIModel, completion_model:OpenAIModel):

        self.variation_model = variation_model
        self.completion_model = completion_model

    def anonymize_text(text_to_anonymize):

        # Example usage:
        anonymizer = Anonymizer()
        
        anonymized_text = anonymizer.anonymize(text_to_anonymize)

        return anonymized_text

    def process_variations(self, index, variables_list, variables_dict, template):
        # Base case: If index is out of range of variables_list, return
        if index > len(variables_list) - 1:

            self.output_array.append(variables_dict.copy()) 

            return 

        # Extract variable names
        variable_name = variables_list[index]

        # Outer loop
        variations = self.generate_variation(variable_name=variable_name, 
                                        variables_dict=variables_dict, 
                                        template=template)["variations"]

        for variation_value in variations:
            # Update dictionary
            variables_dict[variable_name] = variation_value

            # Reset subsequent dictionary keys to => need to have ordered keys => python 3.8""
            for i in range(index + 1, len(variables_list)):
                variables_dict[variables_list[i]] = ""

            self.process_variations(index + 1, variables_list, variables_dict, template=template)



    def generate_variation(self, variable_name:str, variables_dict:dict, template:Template):
        
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
            
            variation_completion = self.variation_model.ask_chat_gpt(system_prompt="No verbose.",
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


    # Initialize the template manager and retrieve the template
    def generate_data(self, template:Template, output_path:str):

        result = []

        # Extracting structures and variables from the template
        prompt = template.prompt 
        completion = template.completion

        prompt_variables = findall(r'\{(.*?)\}', prompt)
        completion_variables = findall(r'\{(.*?)\}', completion)

        prompt_var_dict = {var: "" for var in prompt_variables}

        self.process_variations(index=0, 
                            variables_list=prompt_variables, 
                            variables_dict=prompt_var_dict, 
                            template=template)

        # Loading files outside loops to reduce overhead
        evol_prompt_template = load_file(path="files/evol_instruct.txt")
        completion_prompt_template = load_file(path="files/completion.txt")

        # Always ask to generate X variations for the same prompt
        evol_number = template.prompt_variation_number
        
        for param in self.output_array:
            
            generated_prompt = prompt.format(**param)
            
            # Formatting the reference prompt
            reference_prompt = evol_prompt_template.format(number_of_prompts=str(evol_number), prompt=generated_prompt)
            
            # Generate diversified prompts 
            prompt_diversified_response = self.variation_model.ask_chat_gpt(system_prompt="Answer as a valid JSON like {\"prompts\": [\"XXXX\", \"YYYY\"]}",
                                                        user_prompt=reference_prompt, 
                                                        max_tokens=512, 
                                                        temperature=1,
                                                        json_mode=True)
            
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

                    completion_content = self.completion_model.ask_instruct_gpt(prompt=completion_prompt,
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
    
    def generate_variable(self, variable_name:str, prompt_text:str):
            variations = []
            for _ in range(2):  # Generate 2 variations
                instruction = f"Generate a unique {variable_name} based on the context:\n\n{prompt_text}\n\n{variable_name}:"
                generated_value = self.variation_model.ask_chat_gpt(system_prompt=instruction, user_prompt="", max_tokens=60, temperature=1)
                variations.append(generated_value.strip())
            return variations


    def recursive_variation_generation(self, template:str, variables:list, current_variation_dict:dict):
        if not variables:
            # No more variables to process, generate final variation
            return [template.format(**current_variation_dict)]

        # Get the next variable
        next_var = variables[0]
        remaining_variables = variables[1:]

        # Generate variations for the next variable
        # We create a prompt that includes all the previously generated values
        formatted_template = template.format(**{var: current_variation_dict.get(var, f'{{{var}}}') for var in re.findall(r'\{(.*?)\}', template)})
        current_prompt = formatted_template.split(f'{{{next_var}}}')[0] + f'{{{next_var}}}'  # Keep only the text up to the current variable

        variations = self.generate_variable(variable_name=next_var, prompt_text=current_prompt)

        # Recursively generate combinations for the rest of the variables
        combinations = []

        for variation in variations:
            # Update the current variations dictionary with the new variation
            updated_variation_dict = current_variation_dict.copy()
            updated_variation_dict[next_var] = variation

            # Recursively process the remaining variables
            combinations += self.recursive_variation_generation(
                template,
                remaining_variables,
                updated_variation_dict
            )

        return combinations
    
    def generate_info(self, template:str):
        # Extract variable names
        pattern = re.compile(r'\{(.*?)\}')
        variables = pattern.findall(template)
        # Start the recursive generation process with an empty dictionary for current variations
        return self.recursive_variation_generation(template, variables, {})

    def generate_variation_with_same_structure(self, text:str):
        
        all_combinations = self.generate_info(text)
        
        return all_combinations
        
