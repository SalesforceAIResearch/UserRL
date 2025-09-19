from setuptools import setup, find_packages

setup(
    name="travelgym",
    version="1.0.0",
    description="A Gymnasium environment for travel planning preference elicitation simulation using LLMs",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "numpy",
        "openai",
        "pyyaml",
    ],
    include_package_data=True,
)
