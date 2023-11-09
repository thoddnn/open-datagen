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
from opendatagen.data_generator import DataGenerator
from opendatagen.model import LLM
from opendatagen.template import Template, Variable

variation_model = LLM.load_chat.GPT_35_TURBO_CHAT 
completion_model = LLM.load_instruct.GPT_35_TURBO_INSTRUCT

# Create the custom template using the Pydantic models
user_template = Template(
    description="Custom template for Python exercises",
    prompt="Python exercice statement: {python_exercice_statement}",
    completion="Answer:\n{python_code}",
    prompt_variation_number=1,
    prompt_variables={
        "python_exercice_statement": Variable(
            name="Python exercice statement",
            temperature=1,
            max_tokens=120,
            generation_number=10
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

generator = DataGenerator(template=user_template, variation_model=variation_model, completion_model=completion_model)

data = generator.generate_data(output_path="output.csv")

print(data)
```

This code will generate a dataset of 5 medium-level Python exercises/answers formatted as you asked for.

### Predefined Templates:

```python
from opendatagen.data_generator import DataGenerator
from opendatagen.model import LLM
from opendatagen.template import TemplateManager, TemplateName

variation_model = LLM.load_chat.GPT_35_TURBO_CHAT
completion_model = LLM.load_instruct.GPT_35_TURBO_INSTRUCT

manager = TemplateManager()
template = manager.get_template(TemplateName.PRODUCT_REVIEW)

generator = DataGenerator(template=template, variation_model=variation_model, completion_model=completion_model)

data = generator.generate_data(output_path="output.csv")

print(data)
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

## Connect 

Reach us on Twitter: [@thoddnn](https://twitter.com/thoddnn).