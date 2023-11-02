# ⬜️ Open Datagen ⬜️

**Open Datagen**, a steerable data generation system for ML models training.

## Features

- Generate data in the format you want
- Create custom templates with Pydantic models
- Use predefined templates

## Installation

```bash
pip install --upgrade opendatagen
```

### Setting up the OpenAI API key

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

## Usage

Example: If you want to train a small model to write great python code

```python
from opendatagen.data_manager import Template, Variable, generate_data

# Example: Defining a custom template to generate 5 medium-level Python exercises
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

# Generate your data
data = generate_data(template=user_template, output_path="output.csv")
```

This code will generate a dataset of 5 medium-level Python exercises.

### Predefined Templates:

```python
from opendatagen.data_manager import TemplateManager, Template, Variable, generate_data

# Load templates from template.json 
manager = TemplateManager()
# Or load your own template JSON 
#manager = TemplateManager(filename="custom_template.json")
template = manager.get_template(TemplateName.PRODUCT_REVIEW)

if template:
    data = generate_data(template=template, output_path="output.csv")
```

You can find the templates in the [template.json](https://github.com/thoddnn/open-datagen/blob/main/opendatagen/files/template.json) file.

## Roadmap 

- Enhance completion quality with sources like Internet, local files, and vector databases
- Augment and replicate sourced data
- Ensure data anonymity & open-source model support
- Future releases to support multimodal data
  
## Note 

Please note that `opendatagen` is initially powered by OpenAI's models. Be aware of potential biases and use the `start_with` and `note` field to guide outputs.

## Acknowledgements

We would like to express our gratitude to the following open source projects and individuals that have inspired and helped us:

- **Textbook Generation** by [VikParuchuri](https://github.com/VikParuchuri/textbook_quality)
  
- **Evol-Instruct Paper** ([Read the paper](https://arxiv.org/abs/2306.08568)) by [WizardLM_AI](https://twitter.com/WizardLM_AI)
  
- **GPT-LLM-Trainer** by [mattshumer_](https://twitter.com/mattshumer_) available at [GitHub](https://github.com/mshumer/gpt-llm-trainer)

## Connect 

Reach us on Twitter: [@thoddnn](https://twitter.com/thoddnn).