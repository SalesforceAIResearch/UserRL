from setuptools import setup, find_packages

setup(
    name="turtlegym",
    version="1.0.0",
    description="A Gymnasium environment for Turtle Soup lateral thinking puzzle games",
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
