from setuptools import setup, find_packages

setup(
    name="process_data",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
    ],
    author="Elvis Casco",
    author_email="elvis.casco@bse.edu",
    description="A package for diabetes prediction using machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)