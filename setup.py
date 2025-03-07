from codecs import open
from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="PyMADREQ",
    version="0.1",
    author="Ali A. Eftekhari",
    author_email="e.eftekhari@gmail.com",
    description="A finite volume solver for multicomponent advection-diffusion-reaction equation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simulkade/PyMADREQ",
    license="MIT",
    packages=['pymadreq'],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: under development - unstable",
        "Intended Audience :: Research",
        "License :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
)