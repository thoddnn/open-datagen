from setuptools import setup, find_packages

setup(
    name='open-datagen',
    version='0.0.2',
    packages=find_packages(),
    include_package_data=True,  # This will include non-python files like our .txt and .json files
    install_requires=[
        # Add any required packages here
    ],
)
