from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alfworldgym",
    version="1.0.0",
    author="AlfworldGym Team",
    author_email="",
    description="A Gymnasium environment for alfworld household task completion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/alfworldgym",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Puzzle Games",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "alfworld",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    include_package_data=True,
) 