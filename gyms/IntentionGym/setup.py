from setuptools import setup, find_packages

setup(
    name="intentiongym",
    version="1.0.0",
    description="A Gymnasium environment for intention guessing simulation using LLMs",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "pyyaml>=6.0",
    ],
    include_package_data=True,
)
