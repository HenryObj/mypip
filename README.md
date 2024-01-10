# base.py
👋 Hi there,

This module is to simplify the access to utility functions I use in various codebases. Feel free to use it and suggest improvements 🤝

## Last Update - 10th of January 2024
WIP: Adding the OAI assistants in this package

## What is in this package?
The package is separated in 3 modules:
    1. General utilities functions in 'base.py'
    2. Open AI related functions in 'oai.py'
    3. Internet related functions

TODO => provide some examples in this README to make it more convenient.

**Dependencies**
If not already installed, this module will install the following packages and their dependencies:
* tiktoken - to calculate the number of tokens
* openai - for embedding, chat-gpt-3.5, and chat-gpt-4
* request
* bs4

### Old Updates

November 2023: Added the latest models from Open AI and added a func to extract the conversation in a clean manner even with nested json.

September 2023: (Removed in 2024) We are using Chat-GPT-Instruct when "asking a question" to GPT.

