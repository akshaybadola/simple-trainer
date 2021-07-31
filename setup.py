import os
from setuptools import setup

from simple_trainer import __version__

description = """Deep learning ORChestrator (DORC)
A unified environment for training complex Deep Neural Networks over remote and
distributed systems."""

with open("README.md") as f:
    long_description = f.read()

setup(
    name="simple_trainer",
    version=__version__,
    description=description,
    long_description=long_description,
    url="https://github.com/akshaybadola/simple_trainer",
    author="Akshay Badola",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
    ],
    packages=["simple_trainer"],
    package_data={"simple_trainer": ["py.typed"]},
    include_package_data=True,
    keywords='machine learning deep learning remote management',
    python_requires=">=3.6, <4.0",
    install_requires=[
        # "configargparse==1.2.3",
        # "cycler==0.10.0",
        # quart
        # hypercorn
        # magnum for serverless apparently
        # httpx for what?
        # "flask_docspec @ git+https://github.com/akshaybadola/flask-docspec.git@main",
        # "Flask-Pydantic==0.6.0",
        "common-pyutil>=0.6.1,<=1.0.0",
        "torch>=1.4.0",
        "numpy>=1.17.4",
        "pydantic @ git+https://github.com/akshaybadola/pydantic.git@master",
        "pynvml>=8.0.3",
        "requests>=2.26.0"
    ],
    entry_points={
        'console_scripts': [
            'trainer = trainer:__main__.main',
        ],
    },
)
