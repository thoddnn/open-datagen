# ⬜️ Open Datagen ⬜️

**Open Datagen**, a steerable data generation system for ML models training.

## Features

- Generate high-quality synthetic datasets using simple templates

- Quality enhancement with RAG from Internet and local files

- Data anonymization

- Data evaluation & cleaning agent

- Open-source model support (HuggingFace Inference API)

- (SOON) Data decontamination

- (SOON) Multimodality

## Installation

```bash
pip install --upgrade opendatagen
```

### Setting up your API keys

```bash
export OPENAI_API_KEY='your_openai_api_key' #(using openai>=1.2)
export HUGGINGFACE_API_KEY='your_huggingface_api_key'
export MISTRAL_API_KEY='your_mistral_api_key'
export SERPLY_API_KEY='your_serply_api_key' #Google Search API 
```

## Usage

Example: Generate a low-biased dataset to improve factuality of an LLM.

template.json:
```json
{
    "factuality": {
        "description": "Factuality",
        "prompt": "Given the following text:\n\n'''{wikipedia_content}'''\n\nAnswer to this factually checkable question:\n'''{question}'''.",
        "completion": "Answer: '''{answer}'''. Rate the answer out of 10: {score}",
        "prompt_variation_number": 0,
        "variables": {
            "wikipedia_content": {
                "name": "Wikipedia content",
                "generation_number": 1,
                "get_value_from_huggingface": {
                    "dataset_name": "20220301.en",
                    "dataset_path": "wikipedia",
                    "column_name": "text",
                    "max_tokens": 512
                }
            },
            "question": {
                "name": "Factually checkable question",
                "generation_number": 3,
                "models": [
                    {
                        "openai_chat_model": {
                            "name": "gpt-3.5-turbo-1106",
                            "temperature": 0,
                            "max_tokens": 128
                        }
                    }
                ]
            }, 
            "answer": {
                "name": "Short answer to the question",
                "generation_number": 1,
                "models": [
                    {
                        "openai_instruct_model": {
                            "name": "gpt-3.5-turbo-instruct",
                            "temperature": 0,
                            "max_tokens": 128,
                            "start_with": ["Answer:"]
                        }
                    }
                ]
            
            },
            "score": {
                "name": "Score",
                "generation_number": 1,
                "note": ["You must answer with an integer."],
                "models": [
                    {
                        "openai_chat_model": {
                            "name": "gpt-3.5-turbo-1106",
                            "temperature": 0,
                            "max_tokens": 5
                        }
                    }
                ]
            }
            
        }
    }
}
```

Python code to generate the dataset:
```python
from opendatagen.template import TemplateManager
from opendatagen.data_generator import DataGenerator

output_path = "factuality.csv"
template_name = "factuality"
manager = TemplateManager(template_file_path="template.json")
template = manager.get_template(template_name=template_name)

if template:
    
    generator = DataGenerator(template=template)
    
    data = generator.generate_data(output_path=output_path)
    
    print(data)
```

Using this template you will: 
1) Get text content from the Wikipedia dataset hosted on HuggingFace
2) Generate 3 questions about this content
3) Generate an short answer 
4) Rate the answer

Once the CSV created, you can ask an AI Agent to evaluate and correct your dataset

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
