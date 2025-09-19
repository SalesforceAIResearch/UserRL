from setuptools import setup, find_packages

setup(
    name="searchgym",
    version="1.0.0",
    description="A Gymnasium environment for search-based question answering using Serper API",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "numpy",
        "pyyaml",
    ],
    include_package_data=True,
)
