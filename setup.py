from pathlib import Path

from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ravdl",
    version="0.6",
    license='MIT',
    author="Raven Protocol",
    author_email='kailash@ravenprotocol.com',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ravenprotocol/ravdl',
    keywords='Ravdl, deep learning library, algorithms',
    install_requires=[
        "numpy==1.21.5",
        "terminaltables==3.1.10",
        "onnx==1.12.0",
        "ravop",
        "python-dotenv"
    ]
)
