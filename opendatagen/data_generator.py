
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
from opendatagen.utils import dict_to_string, load_file, write_to_csv, find_strings_in_brackets, find_strings_in_double_brackets
from opendatagen.utils import pydantic_list_to_dict, replace_with_dict
from opendatagen.anonymizer import Anonymizer
from opendatagen.model import OpenAIChatModel, OpenAIInstructModel, OpenAIEmbeddingModel, ModelName, MistralChatModel, LlamaCPPModel, TogetherChatModel, AnyscaleChatModel, UserMessage
from opendatagen.template import Template, Variable, Variations, create_variable_from_name
from opendatagen.utils import function_to_call
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import uuid
import copy 

load_dotenv()

class DataGenerator:

    output_array = []

    def __init__(self, template:Template):

        self.template = template

    def extract_variable_from_string(self, text:str):
        return findall(r'\{\{(.*?)\}\}', text)

    def extract_variable_dict_from_string(self, text:str):
        
        list_of_variables = findall(r'\{\{(.*?)\}\}', text)

        result = {}

        for variable_id, variable in self.template.variables.items():
            
            if variable_id in list_of_variables:
                result[variable_id] = variable

        return result
    
    def anonymize_text(self, text_to_anonymize):

        # Example usage:
        anonymizer = Anonymizer()

        anonymized_text = anonymizer.anonymize(text_to_anonymize)

        return anonymized_text
    
    def contextual_generation(self, variables:list, current_variation_dict:dict, fixed_variables: Dict[str, Variable], completion:str=None, parent_id:str=None):

        # This will be the list to collect all dictionaries
        result = []

        if not variables:
            # No more variables to process, generate final variation
            return [current_variation_dict.copy()]
        
        # Get the next variable
        next_var = variables[0]
        remaining_variables = variables[1:]

        variable = fixed_variables[next_var]

        variations = self.generate_variable(current_variable=variable,
                                            variable_id_string=next_var,
                                            parent_id=parent_id)
        

        
        if variable.junction:

            data:List[str] = [variable.value for variable in variations.values()]
            junction_value = variable.junction.generate(data=data)

            for key, old_value in variations.items():

                temp_value = variations[key]
                temp_value.value = junction_value
                variations[key] = temp_value

            if variable.junction.delete_branch:

                last_key, last_value = list(variations.items())[-1]
                variations.clear()
                variations[last_key] = last_value
            
            variations[key] = temp_value
        
        for id, variation in variations.items():
            # Update the current variations dictionary with the new variation
            updated_variation_dict = current_variation_dict.copy()

            updated_variation_dict[next_var] = variation
            
            # Recursively process the remaining variables
            # and extend the all_variation_dicts list with the results
            result.extend(self.contextual_generation(
                completion=completion,
                variables=remaining_variables,
                current_variation_dict=updated_variation_dict,
                fixed_variables=fixed_variables,
                parent_id=id
            ))
        
        # Return the list of all variation dictionaries generated
        return result
    
    def transform_generated_value(self, current_variable:Variable, value:str, parent_id):

        function_name = current_variable.transform_value.function_name
        from_notebook = current_variable.transform_value.from_notebook
        additional_parameters = current_variable.transform_value.additional_parameters

        param_dict = {}

        if additional_parameters:
            
            for param in additional_parameters:
                
                param_dict[param] = self.template.variables[param].values[parent_id]

        param_dict["value"] = value  

        generated_value = function_to_call(function_name, from_notebook, param_dict)

        return generated_value
    

    def add_variation_value(self, variations:dict, variable_id_string:str, current_variable:Variable, generated_value:str, initial_value:str=None, parent_id:str=None, id:str=None):

        if parent_id:

            if id:
                new_id = id
            else:
                new_id = str(uuid.uuid4())

            new_value = Variations(id=new_id, parent_id=parent_id, value=generated_value, initial_value=initial_value)

            current_variable.values[new_id] = new_value

            variations[new_id] = new_value

            self.template.variables[variable_id_string].values[new_id] = new_value

        else:

            if id:
                id_loop = id
            else:
                id_loop = str(uuid.uuid4())

            new_value = Variations(id=id_loop, parent_id=id_loop, value=generated_value, initial_value=initial_value)

            current_variable.values[id_loop] = new_value

            variations[id_loop] = new_value

            self.template.variables[variable_id_string].values[id_loop] = new_value




                
    def retrieve_value(self, target_key, current_variable_name, parent_id, get_initial_value):
        
        # Get keys in reverse order
        keys_in_reverse = list(self.template.variables.keys())[::-1]
        
        # Find the starting index
        start_index = keys_in_reverse.index(current_variable_name) + 1 if current_variable_name in keys_in_reverse else len(keys_in_reverse)

        def find_value(current_id, keys):
            for key in keys:
                value = self.template.variables[key].values[current_id]
                if value.id == current_id:
                    if key == target_key:
                        if get_initial_value:
                            return value.initial_value
                        else:
                            return value.value 
                    return find_value(value.parent_id, keys[start_index:])
            return None

        # Start the lookup process from the key that comes before the current_variable_name
        return find_value(parent_id, keys_in_reverse[start_index:])



    def generate_variable(self, current_variable:Variable, variable_id_string:str, parent_id:str=None):
        
        generation_number = current_variable.generation_number

        variations = {}

        if current_variable.get_value_from_custom_functions:

            for _ in range(generation_number):

                function_name = current_variable.get_value_from_custom_functions.function_name
                from_notebook = current_variable.get_value_from_custom_functions.from_notebook
                additional_parameters = current_variable.get_value_from_custom_functions.additional_parameters

                param_dict = {}

                if additional_parameters:

                    for param in additional_parameters:

                        param_dict[param] = self.template.variables[param].values[parent_id]
                            
                generated_value = function_to_call(function_name, from_notebook, param_dict)

                if current_variable.transform_value :
                    generated_value = self.transform_generated_value(current_variable=current_variable, value=generated_value, parent_id=parent_id)

                if current_variable.get_value_from_custom_functions.chunking:

                    chunks = current_variable.get_value_from_huggingface.chunking.perform_chunk(text=generated_value)

                    for chunk in chunks: 
                        
                        #add values to variations 
                        self.add_variation_value(variations=variations, 
                                                 variable_id_string=variable_id_string, 
                                                 current_variable=current_variable, 
                                                 generated_value=chunk,
                                                 initial_value=generated_value,
                                                 parent_id=parent_id)
                        
                else: 

                    #add values to variations 
                    self.add_variation_value(variations=variations, 
                                                 variable_id_string=variable_id_string, 
                                                 current_variable=current_variable, 
                                                 generated_value=chunk,
                                                 initial_value=generated_value,
                                                 parent_id=parent_id)

            
            if current_variable.decontamination:
                variations = current_variable.decontamination.decontaminate_variable(variations)

            return variations

        if current_variable.get_value_from_localfile:

            for _ in range(generation_number):

                generated_value = current_variable.get_value_from_localfile.get_content_from_file()

                if current_variable.transform_value :
                    generated_value = self.transform_generated_value(current_variable=current_variable, value=generated_value, parent_id=parent_id)

                if current_variable.get_value_from_huggingface.chunking:
                    chunks = current_variable.get_value_from_huggingface.chunking.perform_chunk(text=generated_value)

                    for chunk in chunks: 
                        
                        #add values to variations 
                        self.add_variation_value(variations=variations, 
                                                 variable_id_string=variable_id_string, 
                                                 current_variable=current_variable, 
                                                 generated_value=chunk,
                                                 initial_value=generated_value,
                                                 parent_id=parent_id)
                        
                else: 

                    #add values to variations 
                    self.add_variation_value(variations=variations, 
                                                 variable_id_string=variable_id_string, 
                                                 current_variable=current_variable, 
                                                 generated_value=chunk,
                                                 initial_value=generated_value,
                                                 parent_id=parent_id)
                    
                
            if current_variable.decontamination:
                variations = current_variable.decontamination.decontaminate_variable(variations)
                    
            return variations
        
        if current_variable.get_value_from_huggingface:

            for _ in range(generation_number):
                
                generated_value = current_variable.get_value_from_huggingface.get_random_value_from_dataset()

                if current_variable.transform_value :
                    generated_value = self.transform_generated_value(current_variable=current_variable, value=generated_value, parent_id=parent_id)

                if current_variable.get_value_from_huggingface.chunking:
                    chunks = current_variable.get_value_from_huggingface.chunking.perform_chunk(text=generated_value)

                    for chunk in chunks: 
                        
                        #add values to variations 
                        self.add_variation_value(variations=variations, 
                                                 variable_id_string=variable_id_string, 
                                                 current_variable=current_variable, 
                                                 generated_value=chunk,
                                                 initial_value=generated_value,
                                                 parent_id=parent_id)
                        
                else: 

                    #add values to variations 
                    self.add_variation_value( variations=variations, 
                                                 variable_id_string=variable_id_string, 
                                                 current_variable=current_variable, 
                                                 generated_value=generated_value,
                                                 initial_value=generated_value,
                                                 parent_id=parent_id)

            if current_variable.decontamination:
                variations = current_variable.decontamination.decontaminate_variable(variations)

            return variations
        
        rag_content = ""
        chosen_models = []
        independent_messages = None 

        current_variable = self.template.variables[variable_id_string]

        if current_variable.source_localfile:
            current_variable.load_local_file()
        elif current_variable.source_localdirectory:
            current_variable.load_local_directory()
        elif current_variable.source_internet:
            current_variable.load_internet_source()
        elif current_variable.source_huggingface:
            current_variable.load_huggingface_dataset()

        for _ in range(generation_number):

            if current_variable.ensure_model_diversity:

                available_models = [model.get_model() for model in current_variable.models if model.get_model() not in chosen_models]

                if available_models:
                    current_model = random.choice(available_models) 
                else:
                    current_model = random.choice(current_variable.models).get_model()

            else:
                
                current_model = random.choice(current_variable.models).get_model()

            chosen_models.append(current_model)

            #Get the variables value in user_prompt from the model object 
            if hasattr(current_model, 'user_prompt') and isinstance(current_model.user_prompt, list):

                initial_messages = pydantic_list_to_dict(lst=current_model.user_prompt, fields=['role', 'content'])
                copy_messages = copy.deepcopy(initial_messages)
                copy_messages_obj = copy.deepcopy(current_model.user_prompt)
                    
                for message in current_model.user_prompt:
                    variables_to_get = find_strings_in_double_brackets(text=message.content)

                    if len(variables_to_get) > 0:
                        
                        temp = {}
                        replace_dict = {}

                        for target_variable_name in variables_to_get:
                            
                            #Manage .value and .initial_value
                            split_result = target_variable_name.split(".")
                            target_name = split_result[0]

                            replace_dict[target_variable_name] = target_name

                            try:
                                get_initial_value = split_result[1]
                                if get_initial_value.lower() == "value":
                                    get_initial_value = False 
                                else:
                                    get_initial_value = True 

                            except IndexError:

                                get_initial_value = False  

                            value = self.retrieve_value(target_key=target_name,
                                                        current_variable_name=variable_id_string,
                                                        parent_id=parent_id,
                                                        get_initial_value=get_initial_value)

                            temp[target_name] = value

                        for old, new in replace_dict.items():
                            message.content = message.content.replace(old, new)

                        message.content = replace_with_dict(message.content, temp)

                        if message.rephraser:
                            message.rephrase()

                
                if current_variable.rag_content:

                    rag_content = f"Here is some context that will help you:\n'''{current_variable.rag_content}\n'''"
                    current_model.user_prompt.append(UserMessage(role="user", content=rag_content))
                
            elif hasattr(current_model, 'user_prompt') and isinstance(current_model.user_prompt, str):
                
                copy_messages_obj = copy.deepcopy(current_model.user_prompt)

                variables_to_get = find_strings_in_double_brackets(text=current_model.user_prompt)
                
                if len(variables_to_get) > 0:
                    
                    temp = {}

                    for target_variable_name in variables_to_get:
                        
                        value = self.retrieve_value(target_key=target_variable_name,
                                                    current_variable_name=variable_id_string,
                                                    parent_id=parent_id,
                                                    get_initial_value=True)
                        
                        temp[target_variable_name] = value

                    current_model.user_prompt = replace_with_dict(current_model.user_prompt, temp)  

                    #if message.rephraser:
                    #   message.rephrase()

                
                if current_variable.rag_content:
                    
                    rag_content = f"Here is some context that will help you:\n'''{current_variable.rag_content}\n'''"
                    current_model.user_prompt.append(UserMessage(role="user", content=rag_content))

            elif hasattr(current_model, 'path') and isinstance(current_model.path, str) :

                copy_messages_obj = copy.deepcopy(current_model.path)

                variables_to_get = find_strings_in_double_brackets(text=current_model.path)
                
                if len(variables_to_get) > 0:
                    
                    temp = {}

                    for target_variable_name in variables_to_get:
                        
                        value = self.retrieve_value(target_key=target_variable_name,
                                                    current_variable_name=variable_id_string,
                                                    parent_id=parent_id,
                                                    get_initial_value=True)
                        
                        temp[target_variable_name] = value

                   
                    current_model.path = replace_with_dict(current_model.path, temp)  
              
                    #if message.rephraser:
                    #   message.rephrase()

                
                if current_variable.rag_content:
                    
                    rag_content = f"Here is some context that will help you:\n'''{current_variable.rag_content}\n'''"
                    current_model.path.append(UserMessage(role="user", content=rag_content))


            else: 
                
                raise ValueError("User prompt is badly formatted")
            

            variation_id = str(uuid.uuid4())
            
            if current_variable.transform_value :
                generated_value = self.transform_generated_value(current_variable=current_variable, value=generated_value, parent_id=parent_id)

                new_value = Variations(id=variation_id,
                                       parent_id=parent_id,
                                       value=generated_value,
                                       initial_value=generated_value,
                                       confidence_score=current_model.confidence_score,
                                       model_used=current_model.name)
                
                current_variable.values[variation_id] = new_value

            

            if current_variable.validator:

                count = 1

                while True:

                    if count > current_variable.validator.retry_number:

                        new_value = Variations(id=variation_id,
                                               parent_id=parent_id,
                                               value=generated_value,
                                               initial_value=generated_value,
                                               error_message=new_message,
                                               confidence_score=current_confidence_score,
                                               model_used=current_model.name)
                        
                        current_variable.values[variation_id] = new_value
                        break
                    
                    generated_value = current_model.ask()

                    if isinstance(current_model, OpenAIChatModel):
                        current_confidence_score = current_model.confidence_score
                    else: 
                        current_confidence_score = {} 

                    self.template.variables[variable_id_string].values[parent_id] = Variations(id=variation_id, 
                                                                                                parent_id=parent_id, 
                                                                                                value=generated_value,
                                                                                                initial_value=generated_value,
                                                                                                confidence_score=current_confidence_score,
                                                                                                model_used=current_model.name)
                    
                    function_name = current_variable.validator.function_name
                    from_notebook = current_variable.validator.from_notebook
                    additional_parameters = current_variable.validator.additional_parameters

                    param_dict = {}

                    if additional_parameters:

                        for param in additional_parameters:

                            param_dict[param] = self.template.variables[param].values[parent_id]
                                
                    isValid, new_message = function_to_call(function_name, from_notebook, param_dict)

                    if isValid:

                        new_value = Variations(id=variation_id,
                                               parent_id=parent_id,
                                               value=generated_value,
                                               initial_value=generated_value,
                                               model_used=current_model.name)

                        current_variable.values[variation_id] = new_value

                        break

                    else:
                        
                        if isinstance(current_model.user_prompt, list):

                            current_model.user_prompt.append(UserMessage(role= "assistant", content = generated_value)) 
                            current_model.user_prompt.append(UserMessage(role= "user", content = new_message))

                        elif isinstance(current_model.user_prompt, str):

                            current_model.user_prompt = f"{current_model.user_prompt}\n\nAssistant:{generated_value}\n\nUser:{new_message}"

                        else:
                            raise ValueError("Unknow type of model")
            

                        current_model.ask()

                        count = count + 1

            else:
                
                generated_value = current_model.ask()
            
            if current_variable.independent_values == False:
            
                if isinstance(current_model.user_prompt, list):
                    
                    current_model.user_prompt.append(UserMessage(role="assistant", content=generated_value))
                    current_model.user_prompt.append(UserMessage(role="user", content="You must generate a new answer that is not similar to the last values. No verbose."))
                    
                elif isinstance(current_model.user_prompt, str):
                    
                    current_model.user_prompt = f"{current_model.user_prompt}\n\nAssistant:{generated_value}\n\nUser:{new_message}"

                else:

                    raise ValueError("Unknow type of model")
                


            #add values to variations 
            self.add_variation_value(id=variation_id,
                                    variations=variations, 
                                    variable_id_string=variable_id_string, 
                                    current_variable=current_variable, 
                                    generated_value=generated_value,
                                    initial_value=generated_value,
                                    parent_id=parent_id)
                

            variations[variation_id] = Variations(id=variation_id,
                                               parent_id=parent_id,
                                               value=generated_value,
                                               initial_value=generated_value,
                                               model_used=current_model.name)
        

            if current_variable.independent_values == True:

                #Reinitialize user_prompt value after generation
                if hasattr(current_model, 'user_prompt'):
                    current_model.user_prompt = copy_messages_obj
                

        return variations

            

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
            error_message = self.template.variables[param].error_message

            if error_message:
                error_str = f"{error_str}\n{error_message}"

        return error_str


    def generate_data(self, output_path:str, output_decontaminated_path:str=None):
        
        # Extracting structures and variables from the template
        prompt = self.template.prompt
        prompt_variables = self.extract_variable_from_string(prompt)
        prompt_fixed_variables = self.extract_variable_dict_from_string(text=self.template.prompt)

        save_as_csv = True

        result = []

        if len(prompt_variables) > 0:
            # Start the recursive generation process with an empty dictionary for current variations
            prompts_parameters = self.contextual_generation(variables=prompt_variables, current_variation_dict={}, fixed_variables=prompt_fixed_variables)

            for p_param in prompts_parameters:

                prompt_param = {}

                for variable_id_string, prompt_variation in p_param.items():

                    prompt_param[variable_id_string] = prompt_variation.value

                    prompt_param[f"error_message_{variable_id_string}"] = prompt_variation.error_message
                    prompt_param[f"confidence_{variable_id_string}"] = str(prompt_variation.confidence_score)
                    prompt_param[f"model_used_{variable_id_string}"] = str(prompt_variation.model_used)

                initial_prompt = prompt.format(**prompt_param)
    
                if save_as_csv:
                    
                    row = {"prompt": initial_prompt}
                    row.update(prompt_param)
                    result.append(row)
                    
                    write_to_csv(result, output_path)

                
        if self.template.decontamination:

            result_after_decontamination = self.template.decontamination.decontaminate(result)
                
            write_to_csv(result_after_decontamination, output_decontaminated_path)
    
            return result, result_after_decontamination

        return result, None 
