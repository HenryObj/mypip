from setuptools import setup, find_packages

setup(
    name="henryobj",
    version="0.1", # need to increment this everytime otherwise Pypi will not accept the new version
    packages=find_packages(),
    install_requires=[
        "openai",
        "tiktoken",
        "requests"
    ],
)

# "somepackage==1.2.3",  # if a specific version is required. Here we do it in a way where any version (latest stable) should work