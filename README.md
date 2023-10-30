# ◼ Open-Datagen ◼

Welcome to the documentation of `open-datagen`, a revolutionary steerable data generation system for model training, proudly brought to you by our start-up. The main goal of this inventive package is to simplify the lives of developers by easing the process of data preparation for fine-tuning Language Model (LLM) or Machine Learning (ML) models. With `open-datagen`, you have the flexibility to generate and augment any data in any desired format.

## 🌐 Installation

Start your journey with `open-datagen` by installing the package in your workspace. You just need to run the following command:

```bash
pip install open-datagen
```

## 🚀 Usage 

`open-datagen` allows you to create custom templates using Pydantic models. Here is a simple guide on how to implement this:

```python
# Define your custom template
user_template = Template(
    description="Custom template for Python exercises",
    prompt="Python exercise: '{python_exercise}'",
    completion="Answer using python:\n---\n{python_code}\n---",
    prompt_variables={
        "python_exercise": Variable(
            name="Python exercise",
            temperature=1,
            max_tokens=50,
            variation_nb=1,
            note="The Python exercise statement must be medium level."
        )
    },
    completion_variables={
        "python_code": Variable(
            name="Python code",
            temperature=0,
            max_tokens=256,
            variation_nb=1
        )
    }
)
# Generate your data
generate_data(template=user_template, output_path="output.csv")
```
Lacking inspiration for your data generation? No worries! `open-datagen` also includes a manager to use predefined templates. Here is how you can use them:

```python
manager = TemplateManager()
template = manager.get_template(template_name=TemplateName.PRODUCT_REVIEW.value)
generate_data(template=template, output_path="output.csv")
```
That's all there is to it! `open-datagen` is here to cheerfully help you breathe life into your data, and make your training smoother. Your cool, yet efficient data companion is now ready to roll. 🎉🚀

Thank you for using `open-datagen`! 

Always keep code cool, but not too much! 😉💻📈
