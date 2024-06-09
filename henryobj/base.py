# Utility functions


from .config import WARNING_UNKNOWN, CODE_ERR_PT, CODE_WARN_PT, CODE_DAILY_PT


from typing import Callable, Any, Union, Optional
from collections import Counter
import pathspec
import datetime
import requests
import inspect
import random
import json
import time
import ast
import os
import re


# *************************************** General Utilities ***************************************

def check_co() -> bool:
    """
    Returns true if we have an internet connection. False otherwise.
    """
    try:
        requests.head("http://google.com")
        return True
    except Exception:
        return False

def check__if_password_safe(password: str) -> bool:
    """
    Checks if a password is composed of regular chars or not.
    """
    if not isinstance(password, str): return False
    return bool(re.match(r'^[a-zA-Z0-9$*&@#%!=+,.:;<>?[\]^_`{|}~"-]+$', password))

def clean_punctuation(text: str) -> str:
    """
    Function to clean a text by removing space before a punctuation sign.
    """
    return re.sub(r'\s([?.!,";:])', r'\1', text)

def clean_text(text: str) -> str:
    """
    Function to clean a text by removing non printable char and removing the excess (double \n and double spaces)
    """
    text = remove_non_printable_light(text)
    text = remove_excess(text)
    return text

def convert_dict_to_text(dictionnary: dict, break_two_lines= False) -> str:
    """
    To have the keys as "title" and the values as "content".
    If break_two_lines is set to True, we separate each block by an additional break line.
    """
    x = '\n\n' if break_two_lines else '\n'
    return x.join([f"{y}\n{value}" for y, value in dictionnary.items()])

def correct_spaces_in_text(text):
    """
    Ensures punctuation marks like ".", "?", "!", ";", and "," are correctly spaced with only one space with the next non-space char. 

    Example: 'Hello,world!How     are you?' would be converted to 'Hello, world! How are you?'
    """
    return re.sub(r"([\.,\?!;])\s*(\S)", r"\1 \2", text)

def count_occurrence_in_text(full_text: str, target_word: str, case_sensitive: bool = False) -> int:
    """
    Counts the occurrences of a target word in a given text.

    Args:
    - full_text (str): The text in which to search for the target word.
    - target_word (str): The word to count.
    - case_sensitive (bool, optional): Whether the search should be case-sensitive. Defaults to False.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    word_counts = Counter(re.findall(rf'\b{re.escape(target_word)}\b', full_text, flags))
    return word_counts[target_word] if case_sensitive else word_counts[target_word.lower()]

def custom_round(num: float, threshold: float = 0.1) -> int:
    """
    Custom rounding function based on a user-defined threshold.
    """
    decimal_part = num % 1  # Get the decimal part more efficiently
    return int(num) + (decimal_part >= threshold)

def extract_dict_from_str(s: str) -> Optional[dict]:
    """
    Attempts to parse a string into a dictionary using ast.literal_eval first,
    then json.loads if the first attempt fails. Logs and returns None if both attempts fail.
    """
    try:
        x = ast.literal_eval(s)
        if isinstance(x, dict):
            return x
    except:
        pass
    try:
        x = json.loads(s)
        if isinstance(x, dict):
            return x
    except Exception as e:
        print(f"Issue with both ast.eval and json.loads. Input Data: {type(s)} * {s}")

def find_sentence_boundary(chunk : str, desired_end : int) -> int:
    """
    Simple function to find the last possible sentence boundary. Returns the position of this character or the length of the text if nothing is found.
    """
    for punct in ('. ', '.', '!', ';'):
        pos = chunk[:desired_end].rfind(punct)
    return len(chunk) if pos == -1 else pos

def is_json(myjson: str) -> bool:
  """
  Returns True if the input is in json format. False otherwise.
  """
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

def generate_unique_integer():
    """
    Returns a random integer. Should be unique because between 0 and 2*32 -1 but still we can check after.
    """
    rand_num = random.randint(0, (1 << 31) - 1)
    return rand_num

def get_content_of_file(file_path : str) -> str:
    """
    Reads and returns the content of a file.
    """
    with open(file_path,"r") as file:
        x = file.read()
    return x

def get_path_repo_of_module():
    """
    Get the repo path of the caller's module.
    """
    return os.path.dirname(get_path_of_module())

def get_path_of_module():
    """
    Returns the path of the current module.
    """
    # Get the frame of the caller's caller (two levels up)
    caller_frame = inspect.stack()[2]
    # Extract the file path from the frame
    module_path = caller_frame.filename
    return os.path.abspath(module_path)

def get_module_name(func: Callable[..., Any]) -> str:
    """
    Given a function, returns the name of the module in which it is defined.
    """
    module = inspect.getmodule(func)
    if module is None:
        return ''
    else:
        return module.__name__.split('.')[-1]

def get_name_of_variable(value) -> Optional[Any]:
    """
    Function which returns the first variable name based on its value.
    """
    for var_name, var_value in globals().items():
        if var_value is value:
            return var_name
    return None

def log_issue(exception: Exception, func: Callable[..., Any], additional_info: str = "") -> None:
    """
    Logs an issue. Can be called anywhere and will display an error message showing the module, the function, the exception and if specified, the additional info.

    Args:
        exception (Exception): The exception that was raised.
        func (Callable[..., Any]): The function in which the exception occurred.
        additional_info (str): Any additional information to log. Default is an empty string.

    Returns:
        None
    """
    now = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
    if hasattr(func, '__name__'):
        function_name = func.__name__
        module_name = get_module_name(func)
    else:
        function_name = func if isinstance(func, str) else WARNING_UNKNOWN
        print(f"🟡 What is this function? {func} * {type(func)}")
        try:
            module_name = get_module_name(func)
        except:
            module_name = "Couldn't get the module name"
    additional = f"""
    ****************************************
    Additional Info: 
    {additional_info}
    ****************************************""" if additional_info else ""
    print(f"""
    ----------------------------------------------------------------
    🚨 ERROR 🚨
    Occurred: {now}
    Module: {module_name} | Function: {function_name}
    Exception: {exception}{additional}
    ----------------------------------------------------------------
    """)

def log_warning(warning:str, func: Callable[..., Any], additional_info: str = "") -> None:
    """
    Logs a warning. Less visible in the console than the log issue but works similarly.

    Args:
        warning: The warning message that was raised.
        func (Callable[..., Any]): The function in which the exception occurred.
        additional_info (str): Any additional information to log. Default is an empty string.

    Returns:
        None
    """
    now = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
    if hasattr(func, '__name__'):
        function_name = func.__name__
        module_name = get_module_name(func)
    else:
        function_name = func if isinstance(func, str) else WARNING_UNKNOWN
        print(f"🟡 What is this function? {func} * {type(func)}")
        try:
            module_name = get_module_name(func)
        except:
            module_name = "Couldn't get the module name"
    additional = f"""
    ****************************************
    Additional Info: 
    {additional_info}
    ****************************************""" if additional_info else ""
    print(f"""
    ----------------------------------------------------------------
    👋 Warning 🟠
    Occurred: {now}
    Module: {module_name} | Function: {function_name}
    Warning message: {warning}{additional}
    ----------------------------------------------------------------
    """)

def log_papertrail(exception: Exception, func: Callable[..., Any], log_level: str, icon: str, additional_info: str = "") -> None:
    """
    Base logging function for Papertrail.
    """
    now = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
    if hasattr(func, '__name__'):
        function_name = func.__name__
        module_name = get_module_name(func)
    else:
        function_name = func if isinstance(func, str) else WARNING_UNKNOWN
        try:
            module_name = get_module_name(func)
        except Exception:
            module_name = "Couldn't get the module name"
    additional = additional_info.replace("\n", " ") if additional_info else ""
    print(f"{icon} {log_level} ** | {function_name} in {module_name} | {exception} ** {additional} | {now} | END")

def log_issue_papertrail(exception: Exception, func: Callable[..., Any], additional_info: str = "") -> None:
    """
    Logging function for Papertrail for Errors (Hourly).
    """
    log_papertrail(exception, func, CODE_ERR_PT, "🚨", additional_info)

def log_warning_papertrail(exception: Exception, func: Callable[..., Any], additional_info: str = "") -> None:
    """
    Logging function for Papertrail for warnings (Hourly).
    """
    log_papertrail(exception, func, CODE_WARN_PT, "🟠", additional_info)

def log_work_papertrail(exception: Exception, func: Callable[..., Any], additional_info: str = "") -> None:
    """
    Logging function for Papertrail for elements that need to be reviewed daily.
    """
    log_papertrail(exception, func, CODE_DAILY_PT, "🟡", additional_info)


def print_style(message, color="blue", bold=False):
    """
    To print a message in a specific color.
    Check the function for the list of available colors.
    """
    bold_code = "\033[1m" if bold else ""
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "light_gray": "\033[37m",
        "dark_gray": "\033[90m",
        "light_red": "\033[91m",
        "light_green": "\033[92m",
        "light_yellow": "\033[93m",
        "light_blue": "\033[94m",
        "light_magenta": "\033[95m",
        "light_cyan": "\033[96m",
        "white": "\033[97m",
        "end": "\033[0m",  # Resets the color/style
    }
    print(f"{bold_code}{colors.get(color, colors['green'])}{message}{colors['end']}")

# local tests
def lprint(*args: Any):
    """
    Custom print function to display that things are well at this particular line number.

    If arguments are passed, they are printed in the format: "At line {line_number} we have: {args}"
    """
    caller_frame = inspect.stack()[1][0]
    line_number = caller_frame.f_lineno
    if not bool(len(args)):
        print(line_number, " - Still good")
    else:
        print(f"Line {line_number}: {args}")

# local tests
def fprint(func: Callable[..., Any], additional_info: str = ""):
    """
    Custom print function to display what is going on a given line of a given module. Used for verbose.
    """
    caller_frame = inspect.stack()[1][0]
    line_number = caller_frame.f_lineno
    module_name = get_module_name(func)
    print(f"** LOG ** {line_number} of {module_name} ** INFO: {additional_info}")
    
def perf(function: Callable[..., Any]):
    """
    To be used as a decorator to a function to display the time to run the said function.
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # Start timing here
        res = function(*args, **kwargs)
        end = time.perf_counter()  # End timing here
        duration = round((end - start), 2)
        print(f"{function.__name__} done in {duration} seconds")
        return res
    return wrapper

def print_dir_structure(startpath: str, include_dot_contents: bool = False, use_pipes: bool = True, save_to_file: bool = False, output_file: str = 'dir_structure.txt'):
    """
    Prints or saves the directory and its content, excluding files and directories specified in .gitignore and __pycache__.
    Optionally includes the content of directories starting with '.' based on 'include_dot_contents', formats output with pipes
    and vertical bars if 'use_pipes' is True, and saves output to a file if 'save_to_file' is True.

    Just run "print_dir_structure(".")
    """
    output_lines = []
    gitignore_spec = read_gitignore(startpath)

    for root, dirs, files in os.walk(startpath, topdown=True):
        # Filter out directories as before
        dirs[:] = [d for d in dirs if d != '__pycache__' and not (d.startswith('.') and not include_dot_contents) and not (gitignore_spec and gitignore_spec.match_file(os.path.relpath(os.path.join(root, d), startpath)))]

        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level if use_pipes else ' ' * 4 * level
        branch = '└───' if use_pipes else ''
        subindent = '│   ' * (level + 1) if use_pipes else ' ' * 4 * (level + 1)
        
        relative_root = os.path.relpath(root, startpath)
        if not relative_root.startswith('.') or include_dot_contents or relative_root == '.':
            line = f"{indent}{branch}{os.path.basename(root)}/"
            output_lines.append(line)
        
        for f in files:
            if (not f.startswith('.') or include_dot_contents) and not (gitignore_spec and gitignore_spec.match_file(os.path.relpath(os.path.join(root, f), startpath))):
                line = f"{subindent}{f}"
                output_lines.append(line)
    # Printing or saving to file
    if save_to_file:
        with open(output_file, 'w') as file:
            file.write('\n'.join(output_lines))
    else:
        for line in output_lines:
            print(line)

def read_gitignore(repository_path: str):
    """
    Read the .gitignore file in the given directory and return a PathSpec object.
    """
    gitignore_path = os.path.join(repository_path, '.gitignore')
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, 'r') as file:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', file)
        return spec
    return None

def remove_break_lines(text: str) -> str:
    """
    Replaces all occurrences of double spaces and newline characters ('\n') with a single space.
    """
    jump = '\n'
    double_space = '  '
    while jump in text:
        text = text.replace(jump, ' ')
    while double_space in text:
        text = text.replace(double_space, ' ')
    return text

def remove_jump_double_punc(text: str) -> str:
    """
    Removes all '\n' and '..' for the function to analyze sentiments.
    """
    jump = '\n'
    text = text.replace(jump,'')
    double = '..'
    while double in text:
        text = text.replace(double,'.')
    return text

def remove_excess(text: str) -> str:
    """
    Replaces all occurrences of double newlines ('\n\n') and double spaces with single newline and space, respectively.
    """
    double_jump = '\n\n'
    double_space = '  '
    while double_jump in text:
        text = text.replace(double_jump, '\n')
    while double_space in text:
        text = text.replace(double_space, ' ')
    return text

def remove_non_printable(text :str) -> str:
    """
    Strong cleaner which removes non-ASCII characters from the input text.
    """
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text) # removes non printable char
    y = text.split()
    z = [el for el in y if all(ord(e) < 128 for e in el)]
    return ' '.join(z)

def remove_non_printable_light(text: str) -> str:
    """
    Light cleaner to remove non-printable characters from the input text. Used in the clean_text()
    """
    return ''.join(char for char in text if char.isprintable() or char.isspace())

def remove_punctuation(text: str) -> str:
    """
    Light cleaner using regex to remove punctuation from a text.
    """
    return re.sub(r'[^\w\s]', '', text) 

def safe_json_load(s: str):
    """
    Attempts to correct improperly escaped sequences and loads the string into a list of dictionaries using json.loads.
    Will return the original string if it fails.
    """
    control_chars = {'\n':'\\n' , '\t':'\\t' , '\r':'\\r' , '\b':'\\b'}
    for char, escape_seq in control_chars.items():
        s = s.replace(char, escape_seq)
    try:
        return json.loads(s)
    except Exception as e:
        print(f"Failed to decode JSON. Error: {str(e)}")
        return s

def sanitize_json_response(response: str) -> Union[str, bool]:
    """
    Ensures the response has a JSON-like structure.

    Args:
        response (str): The input string to sanitize.

    Returns:
        Union[str, bool]: The sanitized answer if the response is JSON-like; otherwise, False.
    """
    bal1, bal2 = response.find("{"), response.find("}")
    if bal1 < 0 or bal2 < 0: 
        return False
    return response[bal1:bal2+1]

def sanitize_text(text : str) -> str:
    """
    Function to clean the text before processing it in the DB - to avoid some errors due to bad inputs.
    """
    text = text.replace("\x00", "") # Remove NUL characters
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")  # Normalize Unicode characters
    text = text.replace("\u00A0", " ") # Replace non-breaking spaces with regular spaces
    text = re.sub("<[^>]*>", "", text) # Remove HTML tags
    text = " ".join(text.split()) # Replace multiple consecutive spaces with a single space
    return text

def split_into_sentences(text: str) -> list[str]:
    """
    Break down a text into sentences based on sentence boundaries.
    """
    return re.split(r'(?<=[.!?;])\s+|\n', text)

def try_json_loads(s: str) -> Optional[Any]:
    """
    Try / Except around the json loads
    """
    try:
        return json.loads(s)
    except Exception as e:
        log_issue(e, try_json_loads, f"For the string {s}")

def write_locally(content: str, file_name: str, folder_path:str=None, format:str="txt") -> None:
    """
    Writes the given content to a file with the specified name, at the specified folder path, and with the specified format.

    Note: If folder_path is not defined, it will write the content in the dir where the module is run. 
    """
    try:
        if not folder_path: folder_path = get_path_repo_of_module()
        if "." not in file_name: file_name += "." + format
        saving_path = os.path.join(folder_path, file_name)
        with open(os.path.join(folder_path, file_name), 'w') as file:
            file.write(content)
        return saving_path
    except Exception as e:
        log_issue(e, write_locally, f"Input data leading to the error:\n{content}\n####\n{file_name}\n####\n{folder_path}\n####\n{format}\n-----------------------------")

# *************************************************************************************************
# ************************************* Date & Time related ***************************************
# *************************************************************************************************

def ensure_valid_date(date_input: Union[datetime.date, str]) -> Union[datetime.date, None]:
    """
    Ensures the given possible date is a valid date and returns it.
    
    Args:
        date_input: Date in a datetime or in a string in recognizable formats.
        
    Returns:
        datetime.date: Parsed date object, or None if parsing fails.
    """
    if isinstance(date_input, datetime.date,): return date_input
    elif isinstance(date_input, str):
        date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y'] 
        for fmt in date_formats:
            try:
                return datetime.datetime.strptime(date_input, fmt).date()
            except ValueError:
                pass
        log_issue(ValueError(f"'{date_input}' is not in a recognized date format."), ensure_valid_date)
        return None
    else:
       log_issue((f"Error: The type of date_input is {type(date_input)} which is not str or datetime"), ensure_valid_date)
       return None

def format_datetime(datetime):
    """
    Takes a Datetime as an input and returns a string in the format "10-Jan-2022"
    """
    return datetime.strftime('%d-%b-%Y')

def format_timestamp(timestamp: str) -> str:
    """
    Converts a timestamp string to a "10-Jan-2022" format.
    
    Returns original timestamp if parsing fails.
    """
    try:
        dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        return format_datetime(dt)
    except ValueError:
        return timestamp

def get_days_from_date(date_input: Union[datetime.date, str], unit: str = 'days') -> Union[int, None]:
    """
    Calculate the number of days or years since the provided date.
    
    Args:
        date_input (Union[datetime.date, str]): The date from which to count, 
            accepted as either a date object or a string in several formats 
            (e.g., "2022-09-01", "01-09-2022", "09/01/2022").
        unit (str): Determines the unit of the returned value; accepts "days" or "years". 
            Defaults to "days".
            
    Returns:
        int: Time passed since the provided date in the specified unit. If the date is invalid or in the future, returns None.
    """
    date = ensure_valid_date(date_input)
    if date is None: return None        
    today = datetime.date.today()
    if today < date: return None
    delta = today - date
    if unit == 'days': 
        return delta.days
    elif unit == 'years':
        return today.year - date.year - ((today.month, today.day) < (date.month, date.day))
    else:
        log_issue(ValueError(f"'{unit}' is not a recognized time unit."), get_days_from_date, f"Invalid time unit: {unit}")
        return None

def get_now(exact: bool = False) -> str:
    """
    Small function to get the timestamp in string format.
    By default we return the following format: "10_Jan_2023" but if exact is True, we will return 10_Jan_2023_@15h23s33
    """
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, "%d_%b_%Y@%Hh%Ms%S") if exact else datetime.datetime.strftime(now, "%d_%b_%Y")

# *************************************************************

if __name__ == "__main__":
    pass