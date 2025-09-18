from setuptools import setup, find_packages

setup(
    name="functiongym",
    version="1.0.0",
    description="A mathematical function learning environment for reinforcement learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.21.0",
    ],
    include_package_data=True,
)
