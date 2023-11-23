# base.py
👋 Hi there,

This module is to simplify the access to my utility functions. Feel free to use it and suggest improvements 🤝

## Last Update - 23rd of November 2023
Added the latest models from Open AI and added a func to extract the conversation in a clean manner even with nested json.
Not following the best practices but it works 👌

## What is in this package?
The codebase is separated in 4 main sections:
    1. General utilities functions
    2. Date & Time related functions
    3. Internet related functions
    4. ChatGPT related functions

TODO => provide some examples in this README to make it more convenient.

**Dependencies**
If not already installed, this module will install the following packages and their dependencies:
* tiktoken - to calculate the number of tokens
* openai - for embedding, chat-gpt-3.5, chat-gpt-instruct, and chat-gpt-4
* request

### Old Updates
September:
We are now using Chat-GPT-Instruct when "asking a question" to GPT. This is faster and more performant than queriying the Chat Model (3.5 turbo).
Asking a question to GPT4 is still possible.

