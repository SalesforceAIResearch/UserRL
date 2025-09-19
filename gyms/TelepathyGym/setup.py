from setuptools import setup, find_packages

setup(
    name="telepathygym",
    version="1.0.0",
    description="A Gymnasium environment for mind reading games using LLMs",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "openai",
        "numpy",
        "pyyaml",
    ],
    include_package_data=True,
)
