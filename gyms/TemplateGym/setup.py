from setuptools import setup, find_packages

setup(
    name="templategym",
    version="1.0.0",
    description="A Gymnasium environment for [your domain description]",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "openai",
    ],
    include_package_data=True,
)
