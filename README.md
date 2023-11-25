# ⬜️ Open Datagen ⬜️

**Open Datagen**, a steerable data generation system for ML models training.

## Features

- Generate high-quality synthetic datasets using simple templates

- Quality enhancement with RAG from Internet and local files

- Data anonymization 

- Data evaluation & cleaning agent

- (SOON) Open-source model support (+ local inference)

- (SOON) Multimodality

## Installation

```bash
pip install --upgrade opendatagen
```

### Setting up the OpenAI API key (using openai>=1.2)

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

### Setting up the SERPLY API key for Google Search API (optional)

```bash
export SERPLY_API_KEY='your_serply_api_key'
```

## Usage

Example: Generate a dataset of Python exercises using a template

```python
from opendatagen.data_generator import DataGenerator
from opendatagen.model import LLM
from opendatagen.template import Template, Variable

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
            generation_number=10,
            model_name="gpt-3.5-turbo-1106"
        )
    },
    completion_variables={
        "python_code": Variable(
            name="Python code",
            temperature=0,
            max_tokens=256,
            generation_number=1,
            model_name="gpt-4"
        )
    }
)

#Or you can load your templates from a json file
#from opendatagen.template import TemplateManager
#user_template = TemplateManager("files/template.json")
#Note: you can find examples of json at https://github.com/thoddnn/open-datagen/blob/main/opendatagen/files/template.json

generator = DataGenerator(template=user_template)

data = generator.generate_data(output_path="output.csv")

print(data)
```


Once created, you can ask an AI Agent to evaluate and correct your dataset

```python
from opendatagen.agent import DataAgent

agent = DataAgent()
    
agent.run()
```

## Contribution 

We welcome contributions to Open Datagen! Whether you're looking to fix bugs, add templates, new features, or improve documentation, your help is greatly appreciated.
  
## Note 

Please note that `opendatagen` is initially powered by OpenAI's models. Be aware of potential biases and use the `note` field to guide outputs.

## Acknowledgements

We would like to express our gratitude to the following open source projects and individuals that have inspired and helped us:

- **Textbook Generation** by [VikParuchuri](https://github.com/VikParuchuri/textbook_quality)
  
- **Evol-Instruct Paper** ([Read the paper](https://arxiv.org/abs/2306.08568)) by [WizardLM_AI](https://twitter.com/WizardLM_AI)

## Connect 

Reach us on Twitter: [@thoddnn](https://twitter.com/thoddnn).