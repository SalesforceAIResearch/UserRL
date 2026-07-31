from setuptools import setup, find_packages

setup(
    name="swegym",
    version="1.0.0",
    description="A Gymnasium environment for user/coding-agent interaction sessions",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "openai",
        "numpy",
    ],
    include_package_data=True,
    package_data={"swegym": ["data/*.json"]},
)
