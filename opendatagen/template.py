from pydantic import BaseModel, validator, ValidationError
from typing import Optional, List, Dict
from enum import Enum
import os 
import json 
from opendatagen.utils import load_file
from opendatagen.model import OpenAIModel

class Variable(BaseModel):
    name: str
    temperature: float
    max_tokens: int
    generation_number: int
    type: Optional[str] = None # like 'int' in your example
    min_value: Optional[int] = None # constrain integer to be >= 0
    max_value: Optional[int] = None
    start_with: Optional[List[str]] = None
    note: Optional[str] = None
    rag_content: Optional[str] = None

    class Config:
        extra = "forbid"  # This will raise an error for extra fields

class Template(BaseModel):
    description: str
    prompt: str
    completion: str
    prompt_variables: Optional[Dict[str, Variable]] = None
    completion_variables: Optional[Dict[str, Variable]] = None
    prompt_variation_number: Optional[int] = 5  

    class Config:
        extra = "forbid"  # This will raise an error for extra fields

class TemplateName(Enum):
    PRODUCT_REVIEW = "product-review"


class TemplateManager:

    def __init__(self, filename="template.json"):
        self.template_file_path = self.get_template_file_path(filename)
        self.templates = self.load_templates()

    def get_template_file_path(self, filename: str) -> str:
        return os.path.join(os.path.dirname(__file__), 'files', filename)

    def load_templates(self) -> Dict[TemplateName, Template]:
        with open(self.template_file_path, 'r') as file:
            raw_data = json.load(file)

        templates = {}
        for key, data in raw_data.items():
            try:
                template_name = TemplateName(key)
                template = Template(**data)
                templates[template_name] = template
            except ValidationError as e:
                print(f"Error in template {key}: {e}")
            except ValueError:
                print(f"Unknown template name {key}")

        return templates

    def get_template(self, template_name: TemplateName) -> Template:
        return self.templates.get(template_name)

def create_variable_from_name(model:OpenAIModel, variable_name:str) -> Variable:

    prompt = load_file(path="files/variable_generation.txt")

    prompt = prompt.format(variable_name=variable_name)

    completion = model.ask_instruct_gpt(prompt=prompt, temperature=0, max_tokens=30)
    
    return Variable(**completion)




