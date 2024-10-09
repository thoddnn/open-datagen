from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='opendatagen',
    author="Thomas DORDONNE",
    author_email="dordonne.thomas@gmail.com",
    description="Synthetic data generation to improve AI and humans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thoddnn/open-datagen",
    version='0.0.4',
    packages=find_packages(),
    include_package_data=True,  
    install_requires=[
        'openai==1.5.0',  
        'python-dotenv>=0.17.1',
        'numpy>=1.23.4',
        'trafilatura>=0.9.1',
        'requests>=2.29.0',
        'tenacity>=8.2.2',
        'pydantic>=2',
        'spacy>=3',
        'tiktoken>=0.5',
        'PyPDF2>=3',
        'pandas>=2',
        'datasets>=2',
        'mistralai',
        'jsonschema',
        'llama-cpp-python>=0.2.24',
        'openai-whisper',
        'elevenlabs==0.3.0b0',
        'Pillow',
        'torch>=2.2.0',
        'audiocraft',
        'anthropic',
        'bark @ git+https://github.com/suno-ai/bark.git'
    ]
)
