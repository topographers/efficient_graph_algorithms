import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ega",
    version="0.0.1",
    author="Topographers Group",
    author_email="efficient.topographers@gmail.com",
    description="Package for Fast Topological Graph Processing",
    long_description="TODO",
    long_description_content_type="text/markdown",
    url="https://github.com/topographers",
    packages=['ega'],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
