from setuptools import setup, find_packages

setup(
    name="taugym",
    version="1.0.0",
    description="A Gymnasium environment for tau-bench style tool-agent-user interactions",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "numpy",
        "pyyaml",
        "litellm",
    ],
    include_package_data=True,
)
