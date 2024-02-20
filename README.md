# ⬜️ Open Datagen ⬜️

**Open Datagen** is a Data Preparation Tool designed to build Controllable AI Systems

It offers improvements for:

**RAG**: Generate large Q&A datasets to improve your Retrieval strategies.

**Evals**: Create unique, “unseen” datasets to robustly test your models and avoid overfitting.

**Fine-Tuning**: Produce large, low-bias, and high-quality datasets to get better models after the fine-tuning process.

**Guardrails**: Generate red teaming datasets to strengthen the security and robustness of your Generative AI applications against attack.

## Additional Features

- Use external sources to generate high-quality synthetic data (Local files, Hugging Face datasets and Internet)

- Data anonymization 

- Open-source model support + local inference

- Decontamination

- Tree of thought 

- (SOON) No-code dataset generation

- (SOON) Multimodality 

## Installation

```bash
pip install --upgrade opendatagen
```

### Setting up your API keys

```bash
export OPENAI_API_KEY='your_openai_api_key' #(using openai>=1.2)
export MISTRAL_API_KEY='your_mistral_api_key'
export TOGETHER_API_KEY='your_together_api_key'
export ANYSCALE_API_KEY='your_anyscale_api_key'
export ELEVENLABS_API_KEY='your_elevenlabs_api_key'
export SERPLY_API_KEY='your_serply_api_key' #Google Search API 
```

## Usage

Example: Generate a low-biased FAQ dataset based on Wikipedia content

```python
from opendatagen.template import TemplateManager
from opendatagen.data_generator import DataGenerator

output_path = "opendatagen.csv"
template_name = "opendatagen"
manager = TemplateManager(template_file_path="faq_wikipedia.json")
template = manager.get_template(template_name=template_name)

if template:
    
    generator = DataGenerator(template=template)
    
    data, data_decontaminated = generator.generate_data(output_path=output_path, output_decontaminated_path=None)
    
```

where faq_wikipedia.json is [here](opendatagen/examples/faq_wikipedia.json)

## Contribution

We welcome contributions to Open Datagen! Whether you're looking to fix bugs, add templates, new features, or improve documentation, your help is greatly appreciated.

## Acknowledgements

We would like to express our gratitude to the following open source projects and individuals that have inspired and helped us:

- **Textbooks are all you need** ([Read the paper](https://arxiv.org/abs/2306.11644)) 

- **Evol-Instruct Paper** ([Read the paper](https://arxiv.org/abs/2306.08568)) by [WizardLM_AI](https://twitter.com/WizardLM_AI)

- **Textbook Generation** by [VikParuchuri](https://github.com/VikParuchuri/textbook_quality)

## Connect

If you need help for your Generative AI strategy, implementation, and infrastructure, reach us on

Linkedin: [@Thomas](https://linkedin.com/in/thomasdordonne).
Twitter: [@thoddnn](https://twitter.com/thoddnn).
