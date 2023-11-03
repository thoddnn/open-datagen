from setuptools import setup, find_packages

setup(
    name='opendatagen',
    version='0.0.4',
    packages=find_packages(),
    include_package_data=True,  # This will include non-python files like our .txt and .json files
    install_requires=[
        'openai>=0.27.0',  # Replace with the minimum version you've tested
        'python-dotenv>=0.17.1',
        'numpy>=1.23.4',
        'trafilatura>=0.9.1',
        'requests>=2.29.0',
        'tenacity>=8.2.2',
        'pydantic>=1.10.13',
    ],
)
