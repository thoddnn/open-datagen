from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='opendatagen',
    author="Thomas DORDONNE",
    author_email="dordonne.thomas@gmail.com",
    description="Steerable data generation system for model training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thoddnn/open-datagen",
    version='0.0.8',
    packages=find_packages(),
    include_package_data=True,  
    install_requires=[
        'openai>=1.1.1',  
        'python-dotenv>=0.17.1',
        'numpy>=1.23.4',
        'trafilatura>=0.9.1',
        'requests>=2.29.0',
        'tenacity>=8.2.2',
        'pydantic>=2',
        'spacy>=3'
    ],
)
