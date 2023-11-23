from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="henryobj",
    version="0.1.51", # need to increment this everytime otherwise Pypi will not accept the new version
     url='https://github.com/HenryObj/mypip',
    packages=find_packages(),
    install_requires=[
        "openai",
        "tiktoken",
        "requests",
        "bs4"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)

# "somepackage==1.2.3",  # if a specific version is required. Here we do it in a way where any version (latest stable) should work