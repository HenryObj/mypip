# Giving access to preconfigured GPT4 roles and functions.


from .config import MAX_TOKEN_WINDOW_GPT4_TURBO, LARGE_INPUT_THRESHOLD, BUFFER_README_INPUT
from .base import read_gitignore, remove_excess, get_now
from .oai import calculate_token, ask_question_gpt4


from typing import Optional
import threading
import time
import sys
import os



# *************************************** SUPPORT FUNCS *******************************************

def contains_code(file_path: str) -> bool:
    """
    Check if the given file contains code based on its file extension.
    """
    # List of file extensions associated with code - can be increased.
    code_extensions = {'.py', '.js', '.jsx', '.java', '.c', '.cpp', '.cs', '.html', '.css', '.php', '.rb', '.swift', '.go'}
    _, file_extension = os.path.splitext(file_path)
    return file_extension in code_extensions

def joining_and_summarizing_modules(repository_path: str) -> Optional[str]:
    """
    Process the modules in the repository and subdirectories to create the code context.
    """
    gitignore_spec = read_gitignore(repository_path)
    result, _ = process_directory(repository_path, repository_path, "", 0, gitignore_spec)
    return result

def process_directory(root_path: str, current_path: str, result: str, total_token: int, gitignore_spec) -> tuple:
    """
    Recursively process directories and files within the given path, considering .gitignore rules.
    """
    for entry in os.listdir(current_path):
        full_path = os.path.join(current_path, entry)
        relative_path = os.path.relpath(full_path, start=root_path)  # Calculate the relative path

        # Check against .gitignore rules using the relative path
        if gitignore_spec and gitignore_spec.match_file(relative_path):
            continue  # Skip files and directories that match the .gitignore patterns

        if os.path.isdir(full_path):
            result, total_token = process_directory(root_path, full_path, result, total_token, gitignore_spec)
        elif contains_code(full_path):
            print(f"Processing the file {full_path}")
            with open(full_path, "r") as doc:
                content = doc.read()
            file_token_count = calculate_token(content)  # Assume calculate_token is defined
            if total_token + file_token_count < MAX_TOKEN_WINDOW_GPT4_TURBO - BUFFER_README_INPUT:
                result += f"\n### START OF {full_path} ###\n" + content + f"\n### END OF {full_path} ###\n\n"
                total_token += file_token_count
            else:
                print("Repo is too large for a single README. Consider breaking down the content.")
                return result, total_token
    return result, total_token

def progress_indicator(message: str):
    """
    Display a dynamic progress indicator in the console.
    """
    for _ in range(10):
        for phase in [".   ", "..  ", "... "]:
            sys.stdout.write("\r" + message + phase)
            sys.stdout.flush()
            time.sleep(1)
    sys.stdout.write("\r" + " " * (len(message) + 5) + "\r")  # Clear line
    sys.stdout.flush()

# ****************************************** GPT FUNCS ********************************************

def gpt_bugbounty_generator(repository_path: str) -> str:
    """
    Enter the path of the repo and we will return a report of all possible bugs in it.

    Note:
        We query GPT-4 twice to improve the quality. We do not deal with large repo (for now)
    """
    get_code_content = joining_and_summarizing_modules(repository_path)
    # Guard clause
    if not get_code_content: return
    query_message = "Querying GPT 4"
    if calculate_token(get_code_content) > LARGE_INPUT_THRESHOLD:
        query_message = "Querying GPT4. The repo is a large input so this might take some time, please wait"

    # Start a separate thread for the progress indicator
    progress_thread = threading.Thread(target=progress_indicator, args=(query_message,))
    progress_thread.start()

    first_bb_result = ask_question_gpt4(question=get_code_content, role=ROLE_BUG_BOUNTY)
    role_reviewer = generate_role_bug_bounty_reviewer(first_bb_result)
    improved_result = ask_question_gpt4(question=first_bb_result, role=role_reviewer)

    progress_thread.join()  # Wait for the progress indicator to finish
    return improved_result

def gpt_generate_bb_report(repository_path: str, verbose = True) -> None:
    """
    Take the repo path as an input and generate a new BB_Report in the repo. 
    
    Note: 
        We add a timestamp after the BugBounty Report (BB_Report) to avoid overwritting existing file.
    """
    new_readme = gpt_readme_generator(repository_path)
    now = get_now()
    readme_path = os.path.join(repository_path, f"BB_Report_{now}.md")
    with open(readme_path, "w") as file:
        file.write(new_readme)
    if verbose:
        print(f"✅ ReadMe is completed and available here 👉 {readme_path}")

def gpt_readme_generator(repository_path: str) -> str:
    """
    Enter the path of the repo and we will return the content of the ReadMe file.

    Note:
        We query GPT-4 twice to improve the quality. We do not deal with large repo (for now)
    """
    get_code_content = joining_and_summarizing_modules(repository_path)
    # Guard clause
    if not get_code_content: return
    query_message = "Querying GPT 4"
    if calculate_token(get_code_content) > LARGE_INPUT_THRESHOLD:
        query_message = "Querying GPT4. The repo is a large input so this might take some time, please wait"

    # Start a separate thread for the progress indicator
    progress_thread = threading.Thread(target=progress_indicator, args=(query_message,))
    progress_thread.start()

    first_readme_result = ask_question_gpt4(question=get_code_content, role=ROLE_README_GENERATOR)
    role_reviewer = generate_role_readme_reviewer(first_readme_result)
    improved_result = ask_question_gpt4(question=get_code_content, role=role_reviewer)

    progress_thread.join()  # Wait for the progress indicator to finish
    return improved_result

def gpt_generate_readme(repository_path: str, verbose = True) -> None:
    """
    Take the repo path as an input and generate a new README.md in the repo. 
    
    Note: 
        We add a timestamp after the README to avoid overwritting existing file.
    """
    new_readme = gpt_readme_generator(repository_path)
    now = get_now()
    readme_path = os.path.join(repository_path, f"README_{now}.md")
    with open(readme_path, "w") as file:
        file.write(new_readme)
    if verbose:
        print(f"✅ ReadMe is completed and available here 👉 {readme_path}")


# ****************************************** PROMPTS **********************************************
    
ROLE_README_GENERATOR = """
You are the best CTO and README.md writer. You follow best practices, you pay close attention to details and you are highly rigorous.
The user will share a codebase containing multiple modules. Each module starts with "### START OF path_to_the_module ###" and ends with "### END OF path_to_the_module ###".

### Instructions ###
1. Think step by step.
2. Analyze the provided codebase and understand the architecture.
2. For each module:
   - Summarize its purpose and functionality.
   - Identify key functions and describe their roles.
   - Note any dependencies or important interactions with other modules.
3. Compile these insights into a well-structured README document that includes:
   - An overview of the entire codebase.
   - A description of each module, including its purpose, main functions, and interactions.
   - Any additional notes or observations that could aid in understanding or using the codebase effectively.

### Important Notes ###
1. You will be tipped $200 for the best and most comprehensive README.md file.
2. My job depends on the quality of the output so you MUST be exhaustive.
3. Do not give your opinion and ONLY return the full README content with Markdown format, nothing else. 
"""

def generate_role_readme_reviewer(current_readme: str) -> str:
    """
    Returns the role of the README reviewer.
    """
    return remove_excess(f"""
    You are the best CTO and README.md writer. You follow best practices, you pay close attention to details and you are highly rigorous.
    The user will provide the codebase related to the "Current README". The Codebase contains multiple modules. Each module starts with "### START OF path_to_the_module ###" and ends with "### END OF path_to_the_module ###".

    ### Instructions ###
    1. Think step by step.
    2. Analyze the "Current README" content.
    3. Analyze carefully the codebase provided by the user, paying close attention to each module.
    3. For each module:
    - Summarize its purpose and functionality.
    - Identify key functions and describe their roles.
    - Note any dependencies or important interactions with other modules.
    4. Compare these notes with the Current README file to ensure that the Current README file contains:
    - A valid overview of the entire codebase.
    - A description of each module, including its purpose, main functions, and interactions.
    - Any additional notes or observations that could aid in understanding or using the codebase effectively.
    5. Return the final README in a way that is exhaustive, with proper formatting, and detailed explanation.

    ### Current README ###
    {current_readme}
    ### Important Notes ###
    1. You will be tipped $200 for the best and most comprehensive README.md file.
    2. My job depends on the quality of the output so you MUST be exhaustive.
    3. Do not give your opinion and ONLY return the full README content with Markdown format, nothing else.
    """)

ROLE_BUG_BOUNTY = """
You are the best CTO with decades of experiences in computer science. You follow best practices, you pay close attention to details and you are highly rigorous.

$$$ Instructions $$$
1. Think step by step.
2. Analyze the provided codebase, paying close attention to each module.
2. For each module:
   - Note any dependencies or important interactions with other functions.
   - Assess if the code has any logical, typo, or syntax issues. You MUST take note of everything that can lead to a bug.
   - Important: You do NOT care about imports and functions that are mentioned but not defined.
3. After assessing all modules and all interactions between functions, write following Markdown formatting:
   - The correction needed for each function.
   - Any concrete and specific recommendation to make the code more robust.

$$$ Example of Input $$$
*** file app.py ***
// Python code for Fast API endpoint
@app.route("/knowledge/<id_client>")
def knowledge(id_client):
    connection = open_connection() # CON OPEN
    log_visit_in_app(connection, id_client, "knowledge") #logvisit
    credits = get_credits(connection, id_client)
    knowledge = get_all_knowledge(connection, id_client)
    know = []
    if not knowledge.empty:
    for index, row in knowledge.iterrows():
            know.append((row[0], row[1], row[2], row[4]))
    return render_template("knowledge.html", knowledge = know,  processing = False, id_client = id_client, credits = credits)

*** End of file ***

$$$ Expected Output: $$$

## Code Review Report
### `app.py` - `knowledge(id_client)` - Issues:

1. **For Loop Indentation**
```
if not knowledge.empty:
    for index, row in knowledge.iterrows():  
```
2. Connection Closure
- Add `connection.close()`before returning.

$$$ Important Notes $$$
1. You will be tipped $200 for the best and most comprehensive Bug Bounty report.
2. My job depends on the quality of the output so you MUST be exhaustive.
3. You do NOT care about imports and functions that are mentioned but not defined. You assume that every function without a code, works. You ONLY analyse the code provided.
4. Do not give general comments, ONLY specific issues with the way to resolve them. If you don't find any issue, simply say "No issue found".
"""

def generate_role_bug_bounty_reviewer(current_readme: str) -> str:
    """
    Returns the role of the Bug Bounty reviewer.
    """
    return remove_excess(f"""
    You are the best CTO and Bug Bounty Hunter. You specialise in finding code issues and bugs. You follow best practices, you pay close attention to details and you are highly rigorous.

    ### Instructions ###
    1. Think step by step.
    2. Analyze the below Bug Bounty file.
    3. Analyze carefully the codebase provided by the user, paying close attention to each module.
    For each module:
        - Note any dependencies or important interactions with other functions.
        - Assess if the code has any logical, typo, or syntax issues. You MUST take note of everything that can lead to a bug.
        - Important: You do NOT care about imports and functions that are mentioned but not defined.
    4. After assessing all modules and all interactions between functions, write following Markdown formatting:
        - The correction needed for each function.
        - Any concrete and specific recommendation to make the code more robust.

    ### Current Bug Bounty ###
    {current_readme}
    ### Important Notes ###
    1. You will be tipped $200 for the best and most comprehensive Bug Bounty report.
    2. My job depends on the quality of the output so you MUST be exhaustive.
    3. You do NOT care about imports and functions that are mentioned but not defined. You assume that every function without a code, works. You ONLY analyse the code provided.
    4. Do not give general comments, ONLY specific issues with the way to resolve them in Markdown format. If you don't find any issue, simply say "No issue found".
    """)

# *************************************************************************************************
# *************************************************************************************************
    
if __name__ == "__main__":
    pass