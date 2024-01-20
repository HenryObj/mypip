
"""
    @Author:				Henry Obegi <HenryObj>
    @Email:					hobegi@gmail.com
    @Creation:			    Thursday 18th of January 2024
    @LastModif:             
    @Filename:			    gpt.py
    @Purpose                Giving access to preconfigured GPT4 roles and functions.
    @Partof                 PIP package
"""

# ************** IMPORTS ****************
from .oai import *

# ****** PATHS & GLOBAL VARIABLES *******

BUFFER_README_INPUT = 30000

# *************************************************************************************************
# *************************************** SUPPORT FUNCS *******************************************
# *************************************************************************************************

def contains_code(file_path: str) -> bool:
    """
    Check if the given file contains code based on its file extension.
    """
    # List of file extensions associated with code - can be increased.
    code_extensions = {'.py', '.js', '.jsx', '.java', '.c', '.cpp', '.cs', '.html', '.css', '.php', '.rb', '.swift', '.go'}
    _, file_extension = os.path.splitext(file_path)
    return file_extension in code_extensions

def joining_and_summarizing_modules(repository_path: str) -> str:
    """
    Process the modules in the repository and subdirectories to create the code context.
    """
    result, _ = process_directory(repository_path, "", 0)
    return result

def process_directory(directory_path: str, result: str, total_token: int) -> tuple:
    """
    Recursively process directories and files within the given path.
    """
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isdir(full_path):
            result, total_token = process_directory(full_path, result, total_token)
        elif contains_code(full_path):
            with open(full_path, "r") as doc:
                content = doc.read()
            file_token_count = calculate_token(content)
            if total_token + file_token_count < MAX_TOKEN_WINDOW_GPT4_TURBO - BUFFER_README_INPUT:
                result += f"\n### START OF {entry} ###\n" + content + f"\n### END OF {entry} ###\n\n"
                total_token += file_token_count
            else:
                print("Repo is too large for a single README. Consider breaking down the content.")
                return result, total_token
    return result, total_token

# *************************************************************************************************
# ****************************************** GPT FUNCS ********************************************
# *************************************************************************************************

def gpt_readme_generator(repository_path: str) -> str:
    """
    Enter the path of the repo and we will return the content of the ReadMe file.

    Note:
        This is a V1 as such, some features are not working (long repo) and we only query GPT 4 once so quality may be low.
    """
    get_code_content = joining_and_summarizing_modules(repository_path)
    first_readme_result = ask_question_gpt4(question=get_code_content, role=ROLE_README_GENERATOR)
    role_reviewer = generate_role_readme_reviewer(first_readme_result)
    improved_result = ask_question_gpt4(question=first_readme_result, role=role_reviewer)
    return improved_result

def gpt_generate_readme(repository_path: str, verbose = True) -> None:
    """
    Take the repo path as an input and generate a new README.md in the repo. 
    
    Note: 
        We add a timestamp after the README to avoid overwritting existing file.
    """
    new_readme = gpt_readme_generator(repository_path)
    now = get_now()
    with open(os.path.join(repository_path, f"README_{now}.md"), "w") as file:
        file.write(new_readme)
    if verbose: print("✅ README.md file generated")

# *************************************************************************************************
# ****************************************** PROMPTS **********************************************
# *************************************************************************************************
    
ROLE_README_GENERATOR = """
You are the best CTO and README.md writer. You follow best practices, you pay close attention to details and you are highly rigorous.

### Instructions ###
1. Think step by step.
2. Analyze the provided codebase, paying close attention to each module.
2. For each module:
   - Summarize its purpose and functionality.
   - Identify key functions and describe their roles.
   - Note any dependencies or important interactions with other modules.
3. Compile these insights into a well-structured README document that includes:
   - An overview of the entire codebase.
   - A description of each module, including its purpose, main functions, and interactions.
   - Any additional notes or observations that could aid in understanding or using the codebase effectively.

### Example of User Input ###
*** Start of file ***
// JavaScript code for a simple calculator

function add(a, b) {
   return a + b;
}

function subtract(a, b) {
   return a - b;
}

// More functions here...

*** End of file - calculator.js ***
*** Start of file ***
// CSS for styling the calculator

body {
   background-color: #f3f3f3;
}

// More styles here...

*** End of file - styles.css ***

### Expected Output: ###

README File:

- **Overview:**
  - The provided codebase consists of two main modules: a JavaScript file (`calculator.js`) for calculator functionalities and a CSS file (`styles.css`) for styling the calculator interface.

- **Module Descriptions:**
  1. `calculator.js`:
     - **Purpose:** Implements basic calculator functions.
     - **Key Functions:**
       - `key_function(a, b)`: Returns the of `a` and `b`.

  2. `styles.css`:
     - **Purpose:** Provides the styling for the calculator's user interface.
     - **Interactions:** This CSS file is used to style the HTML elements manipulated by `calculator.js`.

### Important Notes ###
1. You will be tipped $200 for the best and most comprehensive README.md file.
2. My job depends on the quality of the output so you MUST be exhaustive.
3. Only return the content of the file with the Markdown format, and nothing else.
"""

def generate_role_readme_reviewer(current_readme: str) -> str:
    """
    Returns the role of the README reviewer.
    """
    return remove_excess(f"""
    You are the best CTO and README.md writer. You follow best practices, you pay close attention to details and you are highly rigorous.

    ### Instructions ###
    1. Think step by step.
    2. Analyze the below README file.
    3. Analyze carefully the codebase provided by the user, paying close attention to each module.
    3. For each module:
    - Summarize its purpose and functionality.
    - Identify key functions and describe their roles.
    - Note any dependencies or important interactions with other modules.
    4. Compare these notes with the current README file to ensure that the README file contains:
    - A valid overview of the entire codebase.
    - A description of each module, including its purpose, main functions, and interactions.
    - Any additional notes or observations that could aid in understanding or using the codebase effectively.
    5. Rewrite the full README content to ensure full exhaustivity, proper formatting, and detailed explanation.

    ### Current README ###
    {current_readme}
    ### Important Notes ###
    1. You will be tipped $200 for the best and most comprehensive README.md file.
    2. My job depends on the quality of the output so you MUST be exhaustive.
    3. Only return the content of the file with the Markdown format, and nothing else.
    """)

# *************************************************************************************************
# *************************************************************************************************
    
if __name__ == "__main__":
    pass
   
# WIP for future functions
'''
# Example of Request
REQUEST_WRITE_TEST = """
Write all the tests functions to ensure that the endpoints are correctly working. 
The tests must be broken down into smaller functions. If data is added to the DB, the data should then be removed from the DB if possible (ex: using the /delete endpoint).
print() statements must be used extensively to log on the console the various step and results of those tests.
Below is an example for the endpoints /training. Obviously, you need to do something more complete that this example and test ALL endpoints by calling one main functions which itself calls many smaller functions.
Provide a perfect code, ready to use, with all typing hints and simple docstrings.
"""

# Example of Request
REQUEST_REFACTOR = """
The file assistant.py is poorly made. It composes code snippets, some are useful, others are not. 
What we want is to refactor all this code to have all functions written as cleanly as oai_upload_file.
Write all functions needed to achieve the above results. Only return the code with the typing hint for each one.
"""
'''
