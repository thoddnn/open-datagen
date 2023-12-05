
from dotenv import load_dotenv
import numpy as np
import time
import random
import re
import json
import requests
from urllib.parse import quote
from re import findall
from typing import Dict, List, Union
from opendatagen.utils import dict_to_string, load_file, write_to_csv, generate_context_from_json, extract_website_details, create_type_message, find_strings_in_brackets
from opendatagen.utils import snake_case_to_title_case, title_case_to_snake_case
from opendatagen.utils import extract_content_from_internet
from opendatagen.anonymizer import Anonymizer
from opendatagen.model import OpenAIChatModel, OpenAIInstructModel, OpenAIEmbeddingModel, ModelName
from opendatagen.template import Template, Variable, Variations, create_variable_from_name
from opendatagen.utils import function_to_call
import uuid

load_dotenv()

class DataGenerator:

    output_array = []
    
    def __init__(self, template:Template):

        self.template = template

    def extract_variable_from_string(self, text:str):
        return findall(r'\{(.*?)\}', text)

    def anonymize_text(self, text_to_anonymize):

        # Example usage:
        anonymizer = Anonymizer()
        
        anonymized_text = anonymizer.anonymize(text_to_anonymize)

        return anonymized_text
    
     
    def generate_prompt_variable(self, prompt_text:str, current_variable:Variable, variable_id_string:str, parent_id:str=None):
        
        generation_number = current_variable.generation_number 
    
        variations = {}

        if current_variable.get_value_from_huggingface:

            for _ in range(generation_number):
                
                generated_value = current_variable.get_value_from_huggingface.get_random_value_from_dataset(max_token=current_variable.max_tokens) 
                
                if parent_id:
                    
                    new_id = str(uuid.uuid4())

                    new_value = Variations(id=new_id, parent_id=parent_id, value=generated_value)

                    current_variable.values[new_id] = new_value

                    self.template.prompt_variables[new_id]
                    
                    variations[new_id] = new_value

                    self.template.prompt_variables[variable_id_string].values[new_id] = new_value

                else:

                    id_loop = str(uuid.uuid4())

                    new_value = Variations(id=id_loop, parent_id=parent_id, value=generated_value)

                    current_variable.values[id_loop] = new_value

                    variations[id_loop] = new_value

                    self.template.prompt_variables[variable_id_string].values[id_loop] = new_value

            return variations
    
        initial_variation_prompt = load_file(path="files/generation.txt")

        context = prompt_text

        temp_variation_prompt = initial_variation_prompt
        
        name = current_variable.name 
        model_name = current_variable.model_name
        temperature = current_variable.temperature
        max_tokens = current_variable.max_tokens
        
        note = "" 

        if current_variable.note:
            note = random.choice(current_variable.note) 

        start_with = current_variable.start_with or ""

        comp_type = current_variable.type or ""

        rag_content = ""

        if current_variable.source_localfile:
            current_variable.load_local_file()
        elif current_variable.source_localdirectory:
            current_variable.load_local_directory()
        elif current_variable.source_internet:
            current_variable.load_internet_source()
        elif current_variable.source_huggingface:
            current_variable.load_huggingface_dataset()

        if current_variable.rag_content:
            rag_content = f"Here are some examples that might help you:\n\n{current_variable.rag_content}"

        type_constraint = ""
        
        if comp_type == "int":
            type_constraint = "The variations must be integer."

     
        min_value = current_variable.min_value
        max_value = current_variable.max_value
        start_options = current_variable.start_with

        type_message = create_type_message(comp_type, min_value, max_value)

        start_with = random.choice(start_options) if start_options and start_options != [""] else ""

        last_values_list = []
        last_values = ""

        variation_model = OpenAIChatModel(model_name=model_name)
        
        for _ in range(generation_number): 

            variation_id = str(uuid.uuid4())

            temp_variation_prompt = initial_variation_prompt.format(
                                                        variable_name=variable_id_string,
                                                        rag_content=rag_content, 
                                                        start_with=start_with,
                                                        last_values=last_values,
                                                        type_constraint = type_constraint,
                                                        note=note,
                                                        context=context)

            if isinstance(variation_model, OpenAIInstructModel):
                generated_value = variation_model.ask(prompt=temp_variation_prompt, max_tokens=max_tokens, temperature=temperature)
            else:
                
                start_messages = [
                        {"role": "system", "content": "No verbose."},
                        {"role": "user", "content": temp_variation_prompt},
                ]

                if current_variable.validator:

                    count = 1

                    while True:

                        if count > current_variable.validator.retry_number:

                            new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value, error_message=new_message)
                            current_variable.values[variation_id] = new_value 
                            break 

                        generated_value = variation_model.ask(max_tokens=max_tokens, 
                                                            temperature=temperature, 
                                                            messages=start_messages, 
                                                            json_mode=current_variable.json_mode,
                                                            seed=current_variable.seed)
                        

                        self.template.prompt_variables[variable_id_string].values[parent_id] = Variations(id=variation_id, parent_id=parent_id, value=generated_value) 

                        function_name = current_variable.validator.function_name
                        from_notebook = current_variable.validator.from_notebook
                        additional_parameters = current_variable.validator.additional_parameters

                        param_dict = {}
                        
                        for param in additional_parameters:

                            param_dict[param] = self.template.prompt_variables[param].values[parent_id]
                        
                        #param_dict["generated_value"] = generated_value
                        
                        isValid, new_message = function_to_call(function_name, from_notebook, param_dict)

                        if isValid:
                            
                            new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value)

                            current_variable.values[variation_id] = new_value

                            break
                             
                        else:
                            
                            start_messages.append({"role": "assistant", "content": generated_value})
                            start_messages.append({"role": "user", "content": new_message})
                            count = count + 1

                else:
                    
                    
                    generated_value = variation_model.ask(max_tokens=max_tokens, 
                                                            temperature=temperature, 
                                                            messages=start_messages, 
                                                            json_mode=current_variable.json_mode,
                                                            seed=current_variable.seed)
                    
                    
                    
                    new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value)
                    current_variable.values[variation_id] = new_value 

            last_values_list.append(generated_value)

            # Create the desired string format if last_values_list is not empty
            if last_values_list:
                last_values = "Generate a content value that is not similar to following values:\n" + "\n".join(last_values_list)
            else:
                last_values = ""
            

            variations[variation_id] = new_value

        return variations

    
    def generate_completion_variable(self, prompt_text:str, completion_text:str, current_variable:Variable, variable_id_string:str, parent_id:str=None):
        
        generation_number = current_variable.generation_number 
    
        variations = {}

        if current_variable.get_value_from_huggingface:

            for _ in range(generation_number):
                
                generated_value = current_variable.get_value_from_huggingface.get_random_value_from_dataset(max_token=current_variable.max_tokens) 
                
                if parent_id:
                    
                    new_id = str(uuid.uuid4())

                    new_value = Variations(id=new_id, parent_id=parent_id, value=generated_value)

                    current_variable.values[new_id] = new_value

                    self.template.completion_variables[new_id]
                    
                    variations[new_id] = new_value

                    self.template.completion_variables[variable_id_string].values[new_id] = new_value

                else:

                    id_loop = str(uuid.uuid4())

                    new_value = Variations(id=id_loop, parent_id=parent_id, value=generated_value)

                    current_variable.values[id_loop] = new_value

                    variations[id_loop] = new_value

                    self.template.completion_variables[variable_id_string].values[id_loop] = new_value

            return variations
    
        initial_variation_prompt = load_file(path="files/completion.txt")

        temp_variation_prompt = initial_variation_prompt

        name = current_variable.name 
        model_name = current_variable.model_name
        temperature = current_variable.temperature
        max_tokens = current_variable.max_tokens
        
        note = "" 

        if current_variable.note:
            note = random.choice(current_variable.note) 

        start_with = current_variable.start_with or ""

        comp_type = current_variable.type or ""

        rag_content = ""

        if current_variable.source_localfile:
            current_variable.load_local_file()
        elif current_variable.source_localdirectory:
            current_variable.load_local_directory()
        elif current_variable.source_internet:
            current_variable.load_internet_source()
        elif current_variable.source_huggingface:
            current_variable.load_huggingface_dataset()

        if current_variable.rag_content:
            rag_content = f"Here are some examples that might help you:\n\n{current_variable.rag_content}"


        type_constraint = ""
        
        if comp_type == "int":
            type_constraint = "The variations must be integer."

     
        min_value = current_variable.min_value
        max_value = current_variable.max_value
        start_options = current_variable.start_with

        type_message = create_type_message(comp_type, min_value, max_value)

        start_with = random.choice(start_options) if start_options and start_options != [""] else ""

        last_values_list = []
        last_values = ""

        completion_model = OpenAIChatModel(model_name=model_name)

        for _ in range(generation_number): 

            variation_id = str(uuid.uuid4())

            temp_variation_prompt = initial_variation_prompt.format(prompt=prompt_text,
                                                                            variable_name=name,
                                                                            completion_type="",
                                                                            completion=completion_text,
                                                                            start_with=start_with,
                                                                            last_values=last_values,
                                                                            type_message=type_message,
                                                                            rag_content=rag_content,
                                                                            note=note)

            if isinstance(completion_model, OpenAIInstructModel):
                generated_value = completion_model.ask(prompt=temp_variation_prompt, max_tokens=max_tokens, temperature=temperature)
            else:
                
                start_messages = [
                        {"role": "system", "content": current_variable.system_prompt},
                        {"role": "user", "content": temp_variation_prompt},
                ]

                if current_variable.validator:

                    count = 1

                    while True:

                        if count > current_variable.validator.retry_number:

                            new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value, error_message=new_message)
                            current_variable.values[variation_id] = new_value 
                            break 

                        generated_value = completion_model.ask(max_tokens=max_tokens, 
                                                            temperature=temperature, 
                                                            messages=start_messages, 
                                                            json_mode=current_variable.json_mode,
                                                            seed=current_variable.seed)
                        

                        self.template.completion_variables[variable_id_string].values[parent_id] = Variations(id=variation_id, parent_id=parent_id, value=generated_value) 

                        function_name = current_variable.validator.function_name
                        from_notebook = current_variable.validator.from_notebook
                        additional_parameters = current_variable.validator.additional_parameters

                        param_dict = {}
                        
                        for param in additional_parameters:

                            param_dict[param] = self.template.completion_variables[param].values[parent_id]
                        
                        #param_dict["generated_value"] = generated_value
                        
                        isValid, new_message = function_to_call(function_name, from_notebook, param_dict)

                        if isValid:
                            
                            new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value)

                            current_variable.values[variation_id] = new_value

                            break
                             
                        else:
                            
                            start_messages.append({"role": "assistant", "content": generated_value})
                            start_messages.append({"role": "user", "content": new_message})
                            count = count + 1

                else:
                    
                    
                    generated_value = completion_model.ask(max_tokens=max_tokens, 
                                                            temperature=temperature, 
                                                            messages=start_messages, 
                                                            json_mode=current_variable.json_mode,
                                                            seed=current_variable.seed)
                    
                    
                    
                    new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value)
                    current_variable.values[variation_id] = new_value 

            last_values_list.append(generated_value)

            # Create the desired string format if last_values_list is not empty
            if last_values_list:
                last_values = "Generate a content value that is not similar to following values:\n" + "\n".join(last_values_list)
            else:
                last_values = ""
            

            variations[variation_id] = new_value

        return variations


    def contextual_completion_generation(self, prompt_text:str, completion:str, variables:list, current_variation_dict:dict, fixed_variables: Dict[str, Variable], parent_id:str=None):
        
        # This will be the list to collect all dictionaries
        result = []

        if not variables:
            # No more variables to process, generate final variation
            return [current_variation_dict.copy()] 

        # Get the next variable
        next_var = variables[0]
        remaining_variables = variables[1:]

        # Generate variations for the next variable
        # We create a prompt that includes all the previously generated values
        formatted_template = completion.format(**{var: current_variation_dict.get(var, f'{{{var}}}').value if hasattr(current_variation_dict.get(var, f'{{{var}}}'), 'value') else current_variation_dict.get(var, f'{{{var}}}') for var in re.findall(r'\{(.*?)\}', completion)})
        current_completion = formatted_template.split(f'{{{next_var}}}')[0] + f'{{{next_var}}}'  # Keep only the text up to the current variable

        #formatted_template = completion.format(**{var: current_variation_dict.get(var, f'{{{var}}}') for var in re.findall(r'\{(.*?)\}', completion)})
        #current_completion = formatted_template.split(f'{{{next_var}}}')[0] + f'{{{next_var}}}'  # Keep only the text up to the current variable
        
        variable = fixed_variables[next_var]

        variations = self.generate_completion_variable(prompt_text=prompt_text, 
                                                       completion_text=current_completion, 
                                                       current_variable=variable, 
                                                       variable_id_string=next_var,
                                                       parent_id=parent_id)

        for id, variation in variations.items():
            # Update the current variations dictionary with the new variation
            updated_variation_dict = current_variation_dict.copy()

            updated_variation_dict[next_var] = variation
           
            # Recursively process the remaining variables
            # and extend the all_variation_dicts list with the results
            result.extend(self.contextual_completion_generation(
                prompt_text,
                completion,
                remaining_variables,
                updated_variation_dict,
                fixed_variables=fixed_variables,
                parent_id=id
            ))

        # Return the list of all variation dictionaries generated
        return result
    
    
    def contextual_prompt_generation(self, prompt:str, variables:list, current_variation_dict:dict, fixed_variables: Dict[str, Variable], parent_id:int = None ):
        
        # This will be the list to collect all dictionaries
        result = []

        if not variables:
            # No more variables to process, generate final variation
            return [current_variation_dict.copy()] 

        # Get the next variable
        next_var = variables[0]
        remaining_variables = variables[1:]

        # Generate variations for the next variable
        # We create a prompt that includes all the previously generated values
        #formatted_template = prompt.format(**{var: current_variation_dict.get(var, f'{{{var}}}') for var in re.findall(r'\{(.*?)\}', prompt)})
        #current_prompt = formatted_template.split(f'{{{next_var}}}')[0] + f'{{{next_var}}}'  # Keep only the text up to the current variable
        formatted_template = prompt.format(**{var: current_variation_dict.get(var, f'{{{var}}}').value if hasattr(current_variation_dict.get(var, f'{{{var}}}'), 'value') else current_variation_dict.get(var, f'{{{var}}}') for var in re.findall(r'\{(.*?)\}', prompt)})
        current_prompt = formatted_template.split(f'{{{next_var}}}')[0] + f'{{{next_var}}}'  # Keep only the text up to the current variable

        variable = fixed_variables[next_var]

        variations = self.generate_prompt_variable(variable_id_string=next_var,
                                                   prompt_text=current_prompt,
                                                   current_variable=variable,
                                                   parent_id=parent_id)

        # Recursively generate combinations for the rest of the variables
        for id, variation in variations.items():

            # Update the current variations dictionary with the new variation
            updated_variation_dict = current_variation_dict.copy()

            updated_variation_dict[next_var] = variation

            # Recursively process the remaining variables
            # and extend the all_variation_dicts list with the results
            result.extend(self.contextual_prompt_generation(
                prompt,
                remaining_variables,
                updated_variation_dict,
                fixed_variables=fixed_variables,
                parent_id=id
            ))

        # Return the list of all variation dictionaries generated
        return result


    def generate_evol_instruct_prompt(self, initial_prompt:str):

        evol_prompt_template = load_file(path="files/evol_instruct.txt")

        evol_instruct_prompt = evol_prompt_template.format(number_of_prompts=str(self.template.prompt_variation_number), prompt=initial_prompt)

        start_messages = [
                        {"role": "system", "content": "Answer as a valid JSON like {\"prompts\": [\"XXXX\", \"YYYY\"]}"},
                        {"role": "user", "content": evol_instruct_prompt},
                ]
        
        evol_instruct_model = OpenAIChatModel(model_name=ModelName.GPT_35_TURBO_CHAT.value)

        diversified_prompt_list = evol_instruct_model.ask(max_tokens=512,
                                                            temperature=1,
                                                            messages=start_messages,
                                                            json_mode=True)
        
        evol_instruct_generated_prompt_list = json.loads(diversified_prompt_list)["prompts"]

        return evol_instruct_generated_prompt_list
    

    def get_completion_error_message(self, params:Dict[str, Variable]):
        
        error_str = ""

        for id, param in params.items():
            
            if param.error_message:
                error_str = f"{error_str}\n{param.error_message}"
    
        return error_str.strip()
          
    def get_prompt_error_message(self, params:dict):
        
        error_str = ""

        for param in params:
            error_message = self.template.prompt_variables[param].error_message

            if error_message:
                error_str = f"{error_str}\n{error_message}"
            
        return error_str

    def generate_data(self, output_path):
            
        # Extracting structures and variables from the template
        prompt = self.template.prompt 
        prompt_variables = self.extract_variable_from_string(prompt) 

        completion = self.template.completion
        completion_variables = self.extract_variable_from_string(completion)

        save_as_csv = True 

        result = []

        if len(prompt_variables) > 0:
            # Start the recursive generation process with an empty dictionary for current variations
            prompts_parameters = self.contextual_prompt_generation(prompt=prompt, variables=prompt_variables, current_variation_dict={}, fixed_variables=self.template.prompt_variables)

            for p_param in prompts_parameters:

                prompt_param = {}

                for variable_id_string, prompt_variation in p_param.items():
                    prompt_param[variable_id_string] = prompt_variation.value
                    prompt_param[f"error_message_{variable_id_string}"] = prompt_variation.error_message 

                initial_prompt = prompt.format(**prompt_param)

                prompt_list = [initial_prompt]

                if self.template.prompt_variation_number > 0:

                    prompt_list = self.generate_evol_instruct_prompt(initial_prompt=initial_prompt)

                #No variables in the prompt 
                for prompt_text in prompt_list[:max(self.template.prompt_variation_number,1)]:
                    
                    completion_parameters = self.contextual_completion_generation(prompt_text=prompt_text, completion=completion, variables=completion_variables, current_variation_dict={}, fixed_variables=self.template.completion_variables)

                    for c_param in completion_parameters:
                        
                        completion_param = {}
                        #completion_error_message = self.get_completion_error_message(params=completion_param)
                        for variable_id_string, variation in c_param.items():
                            completion_param[variable_id_string] = variation.value
                            completion_param[f"error_message_{variable_id_string}"] = variation.error_message 

                        completion_result = completion.format(**completion_param)

                        if save_as_csv:
                            
                            row = {"prompt": prompt, "evol_prompt": prompt_text, "completion": completion_result}
                            row.update(completion_param)
                            result.append(row)
                            
                            write_to_csv(result, output_path)

        else:

            prompt_list = [prompt]

            #No variables in the prompt 
            for prompt_text in prompt_list[:max(self.template.prompt_variation_number,1)]:
                
                completion_parameters = self.contextual_completion_generation(prompt_text=prompt_text, completion=completion, variables=completion_variables, current_variation_dict={}, fixed_variables=self.template.completion_variables)
                
                for param in completion_parameters:
                    
                    completion_param = {}
                    #completion_error_message = self.get_completion_error_message(params=completion_param)
                    for variable_id_string, variation in param.items():
                        completion_param[variable_id_string] = variation.value
                        completion_param[f"error_message_{variable_id_string}"] = variation.error_message 

                    completion_result = completion.format(**completion_param)

                    if save_as_csv:
                        
                        row = {"prompt": prompt, "evol_prompt": prompt_text, "completion": completion_result}
                        row.update(completion_param)
                        result.append(row)
                        
                        write_to_csv(result, output_path)


        return result 

