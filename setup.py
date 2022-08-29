import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TopoGrapher",
    version="0.0.1",
    author="Han Lin",
    author_email="hl3199@columbia.edu",
    description="Package for Fast Topological Graph Processing",
    long_description="TODO",
    long_description_content_type="text/markdown",
    url="https://github.com/topographers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
