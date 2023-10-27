from setuptools import setup, find_packages

setup(
    name='superchargeloop',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,  # This will include non-python files like our .txt and .json files
    install_requires=[
        # Add any required packages here
    ],
)
