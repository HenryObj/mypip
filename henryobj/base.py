"""
    @Author:				Henry Obegi <HenryObj>
    @Email:					hobegi@gmail.com
    @Creation:				Friday 1st of September 2023
    @LastModif:             Wednesday 22nd of November 2023
    @Filename:				base.py
    @Purpose                All the utility functions
    @Partof                 pip package
"""

# ************** IMPORTS ****************
import openai
import os
import requests
import datetime
import inspect
import tiktoken
import json
from typing import Callable, Any, Union, List, Dict, Optional
import re
import time
from urllib.parse import urlparse, urlunparse, quote, unquote
import random
from collections import Counter

# ****** PATHS & GLOBAL VARIABLES *******

OAI_KEY = os.getenv("OAI_API_KEY")
openai.api_key = OAI_KEY

# ****** TOKEN LIMITATIONS
MAX_TOKEN_OUTPUT = 4096
MAX_TOKEN_OUTPUT_DEFAULT = 300
MAX_TOKEN_OUTPUT_DEFAULT_HUGE = 3000

MAX_TOKEN_WINDOW_OLD = 4096 
MAX_TOKEN_WINDOW_GPT4_TURBO = 128000
MAX_TOKEN_WINDOW_GPT35_TURBO = 16385
MAX_TOKEN_WINDOW_GPT4 = 8192

# ****** MODELS
MODEL_GPT4_TURBO = r"gpt-4-1106-preview" #Max 128,000 token context window total with 4,096 output
MODEL_GPT4_STABLE = r"gpt-4" # 8K context window and 4,096 output

MODEL_CHAT = r"gpt-3.5-turbo-1106" # Context 16,385 tokens - Reply 4,096
MODEL_CHAT_STABLE = r"gpt-3.5-turbo"
MODEL_INSTRUCT = r"gpt-3.5-turbo-instruct"

MODEL_EMB = r"text-embedding-ada-002"

# ****** OTHER
OPEN_AI_ISSUE = r"%$144$%" # When OpenAI is down
ERROR_MESSAGE = "An error occurred and was logged"


# *************************************************************************************************
# *************************************** General Utilities ***************************************
# *************************************************************************************************

def clean_punctuation(text: str) -> str:
    '''
    Function to clean a text by removing space before a punctuation sign.
    '''
    return re.sub(r'\s([?.!,";:])', r'\1', text)

def clean_text(text: str) -> str:
    '''
    Function to clean a text by removing non printable char and removing the excess (double \n and double spaces)
    '''
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
    '''
    Ensures punctuation marks like ".", "?", "!", ";", and "," are correctly spaced with only one space with the next non-space char. 

    Example: 'Hello,world!How     are you?' would be converted to 'Hello, world! How are you?'
    '''
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

def find_sentence_boundary(chunk : str, desired_end : int) -> int:
    '''
    Simple function to find the last possible sentence boundary. Returns the position of this character or the length of the text if nothing is found.
    '''
    for punct in ('. ', '.', '!', ';'):
        pos = chunk[:desired_end].rfind(punct)
    return len(chunk) if pos == -1 else pos

def is_json(myjson: str) -> bool:
  '''
  Returns True if the input is in json format. False otherwise.
  '''
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

def generate_unique_integer():
    '''
    Returns a random integer. Should be unique because between 0 and 2*32 -1 but still we can check after.
    '''
    rand_num = random.randint(0, (1 << 31) - 1)
    return rand_num

def get_content_of_file(file_path : str) -> str:
    '''
    Reads and returns the content of a file.
    '''
    with open(file_path,"r") as file:
        x = file.read()
    return x

def get_module_name(func: Callable[..., Any]) -> str:
    '''
    Given a function, returns the name of the module in which it is defined.
    '''
    module = inspect.getmodule(func)
    if module is None:
        return ''
    else:
        return module.__name__.split('.')[-1]

def log_issue(exception: Exception, func: Callable[..., Any], additional_info: str = "") -> None:
    '''
    Logs an issue. Can be called anywhere and will display an error message showing the module, the function, the exception and if specified, the additional info.

    Args:
        exception (Exception): The exception that was raised.
        func (Callable[..., Any]): The function in which the exception occurred.
        additional_info (str): Any additional information to log. Default is an empty string.

    Returns:
        None
    '''
    now = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
    module_name = get_module_name(func)
    print(f" * 🚨 ERROR HO144 * Issue in module {module_name} with {func.__name__} ** Info: {additional_info} ** Exception: {exception} ** When: {now}\n")

# local tests
def lprint(*args: Any):
    '''
    Custom print function to display that things are well at this particular line number.

    If arguments are passed, they are printed in the format: "At line {line_number} we have: {args}"
    '''
    caller_frame = inspect.stack()[1][0]
    line_number = caller_frame.f_lineno
    if not bool(len(args)):
        print(line_number, " - Still good")
    else:
        print(f"Line {line_number}: {args}")

def new_chunk_text(text: str, target_token: int = 200) -> List[str]:
    '''
    Much simpler function to chunk the text in blocks by spliting by sentence. The last chunk might be small.
    '''
    tok_text = calculate_token(text)
    if tok_text < 1.1 * target_token:
        return [text]
    print(f"We need to chunk the text.\nCurrent tokens ~ {tok_text}. Target ~ {target_token}.\nLogically we should get about {custom_round(tok_text/target_token)} chunks")
    sentences = split_into_sentences(text)
    
    aprx = False
    # Spacial case if there is no sentences or less sentences than the desired chunk. If so, we chunk by word.
    if len(sentences) < int(tok_text/target_token) + 1:
        sentences = text.split()
        aprx = True
    token_calculator = calculate_token_aproximatively if aprx else calculate_token

    final_chunks = []
    current_chunk = ""
    current_token_count = 0

    for sentence in sentences:
        sentence_tok = token_calculator(sentence)  # This is a function variable here
        new_token_count =  current_token_count + sentence_tok
        
        # If adding this "sentence" doesn't exceed the limit, add it to the current chunk.
        if new_token_count <= target_token * 1.05:
            current_chunk += sentence + " "
            current_token_count = new_token_count

        # If it does exceed the limit, finalize the current chunk and start a new one keeping the sentence.
        else:
            final_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_token_count = sentence_tok

    if current_chunk.strip():
        final_chunks.append(current_chunk.strip())
    return final_chunks

def perf(function: Callable[..., Any]):
    '''
    To be used as a decorator to a function to display the time to run the said function.
    '''
    start = time.perf_counter()
    def wrapper(*args, **kwargs):
        res = function(*args,**kwargs)
        end = time.perf_counter()
        duration = round((end-start), 2)
        print(f"{function.__name__} done in {duration} seconds")
        return res
    return wrapper

def remove_break_lines(text: str) -> str:
    '''
    Replaces all occurrences of double spaces and newline characters ('\n') with a single space.
    '''
    jump = '\n'
    double_space = '  '
    while jump in text:
        text = text.replace(jump, ' ')
    while double_space in text:
        text = text.replace(double_space, ' ')
    return text

def remove_jump_double_punc(text: str) -> str:
    '''
    Removes all '\n' and '..' for the function to analyze sentiments.
    '''
    jump = '\n'
    text = text.replace(jump,'')
    double = '..'
    while double in text:
        text = text.replace(double,'.')
    return text

def remove_excess(text: str) -> str:
    '''
    Replaces all occurrences of double newlines ('\n\n') and double spaces with single newline and space, respectively.
    '''
    double_jump = '\n\n'
    double_space = '  '
    while double_jump in text:
        text = text.replace(double_jump, '\n')
    while double_space in text:
        text = text.replace(double_space, ' ')
    return text

def remove_non_printable(text :str) -> str:
    '''
    Strong cleaner which removes non-ASCII characters from the input text.
    '''
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text) # removes non printable char
    y = text.split()
    z = [el for el in y if all(ord(e) < 128 for e in el)]
    return ' '.join(z)

def remove_non_printable_light(text: str) -> str:
    '''
    Light cleaner to remove non-printable characters from the input text. Used in the clean_text()
    '''
    return ''.join(char for char in text if char.isprintable() or char.isspace())

def remove_punctuation(text: str) -> str:
    '''
    Light cleaner using regex to remove punctuation from a text.
    '''
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
    '''
    Function to clean the text before processing it in the DB - to avoid some errors due to bad inputs.
    '''
    text = text.replace("\x00", "") # Remove NUL characters
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")  # Normalize Unicode characters
    text = text.replace("\u00A0", " ") # Replace non-breaking spaces with regular spaces
    text = re.sub("<[^>]*>", "", text) # Remove HTML tags
    text = " ".join(text.split()) # Replace multiple consecutive spaces with a single space
    return text

def split_into_sentences(text: str) -> List[str]:
    '''
    Break down a text into sentences based on sentence boundaries.
    '''
    return re.split(r'(?<=[.!?;])\s+|\n', text)

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
    '''
    Takes a Datetime as an input and returns a string in the format "10-Jan-2022"
    '''
    return datetime.strftime('%d-%b-%Y')

def format_timestamp(timestamp: str) -> str:
    '''
    Converts a timestamp string to a "10-Jan-2022" format.
    
    Returns original timestamp if parsing fails.
    '''
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
    '''
    Small function to get the timestamp in string format.
    By default we return the following format: "10_Jan_2023" but if exact is True, we will return 10_Jan_2023_@15h23s33
    '''
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, "%d_%b_%Y@%Hh%Ms%S") if exact else datetime.datetime.strftime(now, "%d_%b_%Y")


# *************************************************************************************************
# *************************************** Internet Related ****************************************
# *************************************************************************************************

def check_co() -> bool:
    '''
    Returns true if we have an internet connection. False otherwise.
    '''
    try:
        requests.head("http://google.com")
        return True
    except Exception:
        return False

def check_valid_url(url):
    '''
    Function which takes a string and return True if the url is valid.
    '''
    try:
        result = urlparse(url)
        if len(result.netloc) <= 1: return False # Checks if the user has put a local file
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def clean_url(url):
    '''
    User-submitted urls might not be perfectly fit to be processed by check_valid_url
    '''
    url = url.strip()
    if not url.startswith('http'):
        url = 'https://' + url
    parsed_url = urlparse(url)

    # Clean the domain by removing any unwanted characters
    cleaned_netloc = re.sub(r'[^a-zA-Z0-9\.\-]', '', parsed_url.netloc)

    # Ensure proper percent-encoding of the path component
    unquoted_path = unquote(parsed_url.path)
    quoted_path = quote(unquoted_path)

    cleaned_url = urlunparse(parsed_url._replace(netloc=cleaned_netloc, path=quoted_path))
    return cleaned_url

def get_local_domain(from_url):
    '''
    Get the local domain from a given URL.
    Will return the same domain for https://chat.openai.com/chat" and https://openai.com/chat".
    '''
    try:
        netloc = urlparse(from_url).netloc
        parts = netloc.split(".")
        if len(parts) > 2:
            domain = parts[-2]
        else:
            domain = parts[0]
        print("URL: ", from_url, " Domain: ", str(domain))
        return str(domain)
    except Exception as e:
        log_issue(e, get_local_domain, f"For {from_url}")

def get_primary_lang_code(lang_data: str) -> str:
    # Split the lang_data by comma and extract the main language code of the first segment
    primary_lang_code = lang_data.split(",")[0].split("-")[0]
    return primary_lang_code

# *************************************************************************************************
# ****************************************** GPT Related ******************************************
# *************************************************************************************************

def add_content_to_chatTable(content: str, role: str, chatTable: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Feeds a chatTable with the new query. Returns the new chatTable.
    Role is either 'assistant' when the AI is answering or 'user' when the user has a question.
    Added a security in case change of name.
    """
    new_chatTable = list(chatTable)
    normalized_role = role.lower()
    if normalized_role in ["user", "client"]:
        new_chatTable.append({"role": "user", "content": content})
    else:
        new_chatTable.append({"role": "assistant", "content": content})
    return new_chatTable

def ask_question_gpt(question: str, role ="", max_tokens=MAX_TOKEN_WINDOW_GPT35_TURBO, verbose = True) -> str:
    """
    Queries OpenAI latest turbo 3.5 Model with a specific question.

    Args:
        question (str): The question to ask the model.
        role (str, optional): As the legacy method would initialize a role, you can use previously defined role which will be part of the prompt.
        max_tokens (int, optional): Maximum number of tokens to be for the completion - Knowing that total cannot exceed 16385 tokens and return is max 4096
        verbose (bool, optional): Will print information in the console. Informations are the token cost and the instructions sent ChatGPT.
    Returns:
        str: The model's reply to the question.

    Note:
        If max_tokens is left to 1000, a print statement will recommand you to adjust it.
    """
    initial_token_usage = calculate_token(role) + calculate_token(question)
    if initial_token_usage > MAX_TOKEN_WINDOW_GPT35_TURBO:
        print("Your input is too large for the query regardless of the max_tokens for the reply.")
        return ""
    elif initial_token_usage + max_tokens > MAX_TOKEN_WINDOW_GPT35_TURBO:
        print(f"Your input + the requested tokens for the answer exceed the maximum amount of {MAX_TOKEN_WINDOW_GPT35_TURBO}.\n Please adjust the max_tokens to a MAXIMUM of {MAX_TOKEN_WINDOW_GPT35_TURBO-initial_token_usage}")
        return ""
    if max_tokens == MAX_TOKEN_OUTPUT_DEFAULT:
        print(f"""\nWarning: You are using default max_tokens of {MAX_TOKEN_OUTPUT_DEFAULT}.\n If you don't need that much, it will be faster and cheaper to reduce the max_tokens.\n
              """)
        max_request = max_tokens - initial_token_usage
    else:
        max_request = max_tokens
    if role == "": instructions = question
    else:
        instructions = f"""You must follow strictly the Role to answer the Question.
        \nRole = {role}
        \n
        Question = {question}
        \nMake sure you take your time to understand the Role and follow the Role before answering the Question. Important: Answer ONLY the Question and nothing else.
        """
        initial_token_usage += 50
    if verbose:
        print(f"Completion ~ {max_tokens} tokens. Request ~ {initial_token_usage} tokens.\nInstructions provided to GPT are:\n{instructions}")
    return request_gpt_instruct(instructions=instructions, max_tokens=max_request)

def ask_question_gpt_instruct(question: str, role ="", max_tokens=MAX_TOKEN_OUTPUT_DEFAULT, verbose = True) -> str:
    """
    Queries OpenAI Instruct Model with a specific question. This had a better performance than getting a chat completion.

    Args:
        question (str): The question to ask the model.
        role (str, optional): As the legacy method would initialize a role, you can use previously defined role which will be part of the prompt.
        max_tokens (int, optional): Maximum number of tokens to be for the completion - Knowing that total cannot exceed 4097
        verbose (bool, optional): Will print information in the console. Informations are the token cost and the instructions sent ChatGPT.
    Returns:
        str: The model's reply to the question.

    Note:
        If max_tokens is left to 1000, a print statement will recommand you to adjust it.
        Edit 8th of Nov - I believe the 3.5 new turbo is better than the instruct
    """
    initial_token_usage = calculate_token(role) + calculate_token(question)
    if initial_token_usage > MAX_TOKEN_WINDOW_GPT35_TURBO:
        print("Your input is too large for the query regardless of the max_tokens for the reply.")
        return ""
    elif initial_token_usage + max_tokens > MAX_TOKEN_WINDOW_GPT35_TURBO:
        print(f"Your input + the requested tokens for the answer exceed the maximum amount of {MAX_TOKEN_WINDOW_GPT35_TURBO}.\n Please adjust the max_tokens to a MAXIMUM of {MAX_TOKEN_WINDOW_GPT35_TURBO-initial_token_usage}")
        return ""
    if max_tokens == MAX_TOKEN_OUTPUT_DEFAULT:
        print(f"""\nWarning: You are using default max_tokens of {MAX_TOKEN_OUTPUT_DEFAULT}.\n If you don't need that much, it will be faster and cheaper to reduce the max_tokens.\n
              """)
        max_request = max_tokens - initial_token_usage
    else:
        max_request = max_tokens
    if role == "": instructions = question
    else:
        instructions = f"""You must follow strictly the Role to answer the Question.
        \nRole = {role}
        \n
        Question = {question}
        \nMake sure you take your time to understand the Role and follow the Role before answering the Question. Important: Answer ONLY the Question and nothing else.
        """
        initial_token_usage += 50
    if verbose:
        print(f"Completion ~ {max_tokens} tokens. Request ~ {initial_token_usage} tokens.\nInstructions provided to GPT are:\n{instructions}")
    return request_gpt_instruct(instructions=instructions, max_tokens=max_request)

def ask_question_gpt4(role: str, question: str, model=MODEL_GPT4_TURBO, max_tokens=MAX_TOKEN_OUTPUT_DEFAULT_HUGE, verbose = False, json_on=False) -> str:
    """
    Queries Chat GPT 4 with a specific question. 

    Args:
        role (str): The system prompt to be initialized in the chat table. How you want ChatGPT to behave.
        question (str): The question to ask the model.
        model (str, optional): The model to use. Defaults to GPT4. Althought it says GPT4, you can use the ChatModel
        max_tokens (int, optional): Maximum number of tokens for the answer. Default is 3000 which is huge.
        verbose (bool, optional): Will print information in the console. Informations are the token cost and the instructions sent ChatGPT.
    Returns:
        str: The model's reply to the question.

    Note:
        If max_tokens is set to 3000, a print statement will prompt you to adjust it.
        Verbose is set to False by default as context is generally very long in GPT4 which flods the console.
    """
    maxi = MAX_TOKEN_WINDOW_GPT4_TURBO if model == MODEL_GPT4_TURBO else MAX_TOKEN_WINDOW_GPT4
    initial_token_usage = calculate_token(role) + calculate_token(question)

    if initial_token_usage > maxi:
        print("Your input is too large for the query regardless of the max_tokens for the reply.")
        return ""
    elif initial_token_usage + max_tokens > maxi:
        print(f"Your input + the requested tokens for the answer exceed the maximum amount of {maxi}.\n Please adjust the max_tokens to a MAXIMUM of {maxi-initial_token_usage}")
        return ""
    if max_tokens == MAX_TOKEN_OUTPUT_DEFAULT_HUGE:
        print(f"""\nWarning: You are using default max_tokens.\n Your default response length is 2500 words. This will likely take a lot of time to process!!\nIf you don't need that much, it will be faster and cheaper to reduce the max_tokens.\n
              """)
    current_chat = initialize_role_in_chatTable(role)
    current_chat = add_content_to_chatTable(question, "user", current_chat)
    if verbose:
        print(f"Completion ~ {max_tokens} tokens. Request ~ {initial_token_usage} tokens.\Context provided to GPT is:\n{current_chat}")
    return request_chatgpt(current_chat, max_tokens=max_tokens, model=model, json_on=False)

def calculate_token(text: str) -> int:
    """
    Calculates the number of tokens for a given text using a specific tokenizer.

    Args:
        text (str): The text to calculate tokens for.

    Returns:
        int: The number of tokens in the text or -1 if there's an error.
    
    Note:
        Uses the tokenizer API and takes approximately 0.13 seconds per query.
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error calculating tokens. Input type: {type(text)}. Exception: {e}")
        return -1

def calculate_token_aproximatively(text: str) -> int:
    '''
    Returns the token cost for a given text input without calling tiktoken.

    2 * Faster than tiktoken but less precise. Will go on the safe side (so real tokens is less)

    Method: A token is about 4 char when it's text but when the char is special, it consumes more token.
    '''
    try:
        nb_words = len(text.split())
        normal, special, asci = 0,0,0
        for char in text:
            if str(char).isalnum():
                normal +=1
            elif str(char).isascii():
                asci +=1
            else:
                special +=1
        res = int(normal/4) + int(asci/2) + 2 * special + 2
        if normal < special + asci:
            return int(1.362 * (res + int(asci/2) +1)) #To be on the safe side
        return int(1.362 * int((res+nb_words)/2))
    except Exception as e:
        log_issue(e,calculate_token_aproximatively,f"The text was {type(text)} and {len(text)}")
        return calculate_token(text)

def change_role_chatTable(previous_chat: List[Dict[str, str]], new_role: str) -> List[Dict[str, str]]:
    '''
    Function to change the role defined at the beginning of a chat with a new role.
    Returns the new chatTable with the system role updated.
    '''
    if previous_chat is None:
        log_issue("Previous_chat is none", change_role_chatTable)
        return [{'role': 'system', 'content': new_role}]
    if not isinstance(previous_chat, list):
        log_issue("Previous_chat is not a list", change_role_chatTable)
        return [{'role': 'system', 'content': new_role}]
    if len(previous_chat) == 0:
        log_issue("Previous_chat is empty", change_role_chatTable)
        return [{'role': 'system', 'content': new_role}]
    new_chat = list(previous_chat)
    if new_chat[0]['role'] == 'system':
        new_chat[0] = {'role': 'system', 'content': new_role}
    else:
        new_chat.insert(0, {'role': 'system', 'content': new_role})
    return new_chat

def check_for_ai_warning(text: str) -> bool:
    """
    Checks if the given text contains phrases that indicate an "AI warning".
    
    Parameters:
    - text (str): The text to be searched for AI warning phrases.
    
    Returns:
    - bool: True if any of the AI warning phrases are found in the text, otherwise False.
    """
    # Updated pattern to include "as a large language model" and remove less specific phrases.
    pattern = r'\b(as an ai|as a large language model|as a chatbot|as a virtual assistant|'\
              r'as a bot|as an artificial intelligence|as an automated|'\
              r'Assistant:|GPT-?\d+|OpenAI)\b'
    
    # re.IGNORECASE makes the search case-insensitive.
    return bool(re.search(pattern, text, re.IGNORECASE))

def check_if_gptconv_format(conversation: List[Dict[str, str]]) -> bool:
    '''
    Checks if the structure of the List matches the GPT conversation format.
    
    Returns:
    - bool: True if valid, False otherwise.
    '''
    for entry in conversation:
        if not set(entry.keys()) == {'role', 'content'}:
            return False
        if not isinstance(entry['role'], str) or not isinstance(entry['content'], str):
            return False
    return True

def embed_text(text: str, max_attempts: int = 3) -> List[float]:
    '''
    Micro function which returns the embedding of one chunk of text or 0 if issue.
    Used for the multi-threading.
    '''
    res = [0]
    if text == "": return res
    attempts = 0
    while attempts < max_attempts:
        try:
            res = openai.Embedding.create(input=text, engine=MODEL_EMB)['data'][0]['embedding']
            return res
        except Exception as e:
            if not check_co():
                print(" Warning: You don't have internet. Embedding will not work. Returning 0.")
                return 0
            attempts += 1
            print(f"OAI Embedding faced the exception {e} at attempt # {attempts} out of 3")
    log_issue(f"No answer despite {max_attempts} attempts", embed_text, "Open AI is down. Returning 0")
    return res

def get_gptconv_readable_format(gpt_conversation: str, system_message: bool = True) -> str:
    """
    Formats a string format GPT conversation (after being extracted from DB) in a human-friendly way.
    If you don't want the system message, mention False. Default, we put it

    Returns:
        - str: The formatted GPT conversation or Error message
    """
    # guard clause
    if not gpt_conversation or not isinstance(gpt_conversation, str): return "Failed to convert to a GPT conversation (not a valid string input)\n"
    try:
        gpt_conv_as_list = []
        # first, we add the system message
        if system_message:
            anchor_1 = gpt_conversation.find('"role": "system", "content":') # len 28 so 30 with the space
            if anchor_1 == -1:
                log_issue("Failed to convert to a GPT conversation", get_gptconv_readable_format, f"No system prompt - wrong format for the input {gpt_conversation}")
                return ERROR_MESSAGE
            anchor_2 = gpt_conversation.find('"role": "user", "content":') # len 26 so 28 with the space
            anchor_2_alt = gpt_conversation.find('"role": "assistant", "content":') # len 31 so 33 with the space
            
            # only the system message
            if anchor_2 == -1: 
                print("Warning: Your GPT conversation only contains the system prompt")
                return f"system: {gpt_conversation[anchor_1+30:-3].strip()}"
            elif 0 < anchor_2 < anchor_2_alt or (0 < anchor_2 and anchor_2_alt == -1):
                gpt_conv_as_list.append(["system", gpt_conversation[anchor_1+30:anchor_2-5].strip()]) # -5 comes from "}, {" of the next element
            # Self-affirmation role
            elif 0 < anchor_2_alt < anchor_2:
                gpt_conv_as_list.append(["system", gpt_conversation[anchor_1+30:anchor_2_alt-5].strip()])
            else:
                log_issue("Failed to convert to a GPT conversation", get_gptconv_readable_format, f"Weird structure for the input {gpt_conversation}")
                return ERROR_MESSAGE
        while True:
            anchor_1 = gpt_conversation.find('"role": "user", "content":') # len 26 so 28 with the space
            anchor_2 = gpt_conversation.find('"role": "assistant", "content":') # len 31 so 33 with the space
            if anchor_1 == -1:
                if anchor_2 == -1:
                    break
                gpt_conv_as_list.append(["assistant", gpt_conversation[anchor_2+33:-3].strip()])
                break
            elif anchor_2 == -1:
                gpt_conv_as_list.append(["user", gpt_conversation[anchor_1+28:-3].strip()])
                break
            else:
                if anchor_1 < anchor_2:
                    # safety because sometimes we have several users in a row
                    anchor_1_safety = gpt_conversation[anchor_1+29:].find('"role": "user", "content":')
                    anchor_2 = anchor_2 if anchor_2-anchor_1 < anchor_1_safety or anchor_1_safety == -1 else anchor_1_safety +34 # 29 + 5. 29 comes from searching the string after anchor_1 + 29. The 5 comes from the difference between anchor_2 (33) and anchor_1 (28)
                    gpt_conv_as_list.append(["user", gpt_conversation[anchor_1+28:anchor_2-5].strip()]) # -5 comes from "}, {"
                    gpt_conversation = gpt_conversation[anchor_2-5:]
                else:
                    # safety because we could have several assistants in a row
                    anchor_2_safety = gpt_conversation[anchor_2+34:].find('"role": "assistant", "content":')
                    anchor_1 = anchor_1 if anchor_1-anchor_2 < anchor_2_safety or anchor_2_safety == -1 else anchor_2_safety + 29 # 34 comes from searching the string after anchor_2 + 34 minus the difference between anchor_1 and anchor 2.
                    gpt_conv_as_list.append(["assistant", gpt_conversation[anchor_2+33:anchor_1-5].strip()])
                    gpt_conversation = gpt_conversation[anchor_1-5:]
        result_string = '\n'.join(': '.join(pair) for pair in gpt_conv_as_list)
        return result_string
    except Exception as e:
        log_issue(e, get_gptconv_readable_format, f"This was the input {gpt_conversation}")
        return ERROR_MESSAGE

def initialize_role_in_chatTable(role_definition: str) -> List[Dict[str, str]]:
    '''
    We need to define how we want our model to perform.
    This function takes this definition as a input and returns it into the chat_table_format.
    '''
    return [{"role":"system", "content":role_definition}]

def print_gpt_models():
    '''
    To list the gpt models provided by OpenAI.
    '''
    response = openai.Model.list() # list all models

    for elem in response["data"]:
        name = elem["id"]
        if "gpt" in name or "embedding" in name: print(name)

def print_gptconv_nicely(gpt_conversation: str) -> None:
    """
    Prints a string format GPT conversation (after being extracted from DB) in a human-friendly way.
    """
    print(get_gptconv_readable_format(gpt_conversation))

# For local tests
def print_len_token_price(file_path_or_text, Embed = False):
    '''
    Basic function to print out the length, the number of token, of a given file or text.
    Chat gpt-3.5-turbo is at $0.002 per 1K token while Embedding is at $0.0004 per 1K tokens. If not specified, we assume it's Chat gpt-3.5-turbo.
    '''
    price = 0.002 if not Embed else 0.0004
    if os.path.isfile(file_path_or_text):
        name = os.path.basename(file_path_or_text)
        with open(file_path_or_text, "r") as file:
            content = file.read()
    elif isinstance(file_path_or_text, str):
        content = file_path_or_text
        name = "Input text"
    else:
        return # to avoid error in case of wrong input
    tok = calculate_token(content)
    out = f"{name}: {len(content)} chars  **  ~ {tok} tokens ** ~ ${round(tok/1000 * price,2)}"
    print(out)

def request_chatgpt(current_chat: list, max_tokens: int, stop_list=False, max_attempts=3, model=MODEL_CHAT, temperature=0, top_p=1, json_on=False):
    """
    Calls the ChatGPT OpenAI completion endpoint with specified parameters.

    Args:
        current_chat (list): The prompt used for the request.
        max_tokens (int): The maximum number of tokens to be used in the context (context = reply + question and role)
        stop_list (bool, optional): Whether to use specific stop tokens. Defaults to False.
        max_attempts (int, optional): Maximum number of retries. Defaults to 3.
        model (str, optional): ChatGPT OpenAI model used for the request. Defaults to 'MODEL_CHAT'.
        temperature (float, optional): Sampling temperature for the response. A value of 0 means deterministic output. Defaults to 0.
        top_p (float, optional): Nucleus sampling parameter, with 1 being 'take the best'. Defaults to 1.
        json (bool, optional): Whether we want to force the output in JSON or not.

    Returns:
        str: The response text or 'OPEN_AI_ISSUE' if an error occurs (e.g., if OpenAI service is down).
    """
    #if model in [MODEL_CHAT, MODEL_GPT4_TURBO]:
    #    response_format = "json_object" if json_on else "text"
    #else:
    #    log_issue("You are using a model which doesn't support JSON object - we depreciated the old models", request_chatgpt)
    #    return ""
    stop = stop_list if (stop_list and len(stop_list) < 4) else ""
    attempts = 0
    valid = False
    #print("Writing the reply for ", current_chat) # Remove in production - to see what is actually fed as a prompt
    while attempts < max_attempts and not valid:
        try:
            response = openai.ChatCompletion.create(
                messages= current_chat,
                temperature=temperature,
                max_tokens= int(max_tokens),
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
                model= model,
                #response_format=  - for now it doesn't work
            )
            rep = response["choices"][0]["message"]["content"]
            rep = rep.strip()
            valid = True
        except Exception as e:
            rep = OPEN_AI_ISSUE
            attempts += 1
            if 'Rate limit reached' in e:
                print(f"Rate limit reached. We will slow down and sleep for 300ms. This was attempt number {attempts}/{max_attempts}")
                time.sleep(0.3)
            else:
                print(f"Error. This is attempt number {attempts}/{max_attempts}. The exception is {e}. Trying again")
    if rep == OPEN_AI_ISSUE and check_co():
        print(f" ** We have an issue with Open AI using the model {model}")
        log_issue(f"No answer despite {max_attempts} attempts", request_chatgpt, "Open AI is down")
    else:
        return rep

def request_gpt_instruct(instructions: str, max_tokens=300, max_attempts=3, temperature=0, top_p=1) -> str:
    '''
    Calls the OpenAI completion endpoint with specified parameters.

    Args:
        instructions (str): The prompt used for the request.
        max_tokens (int): The maximum number of tokens in the reply - defaulted to 300 (200 words)
        max_attempts (int, optional): Maximum number of retries. Defaults to 3.
        temperature (float, optional): Sampling temperature for the response. A value of 0 means deterministic output. Defaults to 0.
        top_p (float, optional): Nucleus sampling parameter, with 1 being 'take the best'. Defaults to 1.

    Returns:
        str: The response text or 'OPEN_AI_ISSUE' if an error occurs (e.g., if OpenAI service is down).
    '''
    attempts = 0
    valid = False
    while attempts < max_attempts and not valid:
        try:
            rep = openai.Completion.create(
                    model = MODEL_INSTRUCT,
                    prompt = instructions,
                    temperature = temperature,
                    max_tokens = max_tokens,
                    top_p =top_p,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            rep = rep["choices"][0]["text"].strip()
            valid = True
        except Exception as e:
            attempts += 1
            rep = OPEN_AI_ISSUE
            if 'Rate limit reached' in e:
                print(f"Rate limit reached. We will slow down and sleep for 300ms. This was attempt number {attempts}/{max_attempts}")
                time.sleep(0.3)
            else:
                print(f"Error. This is attempt number {attempts}/{max_attempts}. The exception is {e}. Trying again")
                rep = OPEN_AI_ISSUE
    if rep == OPEN_AI_ISSUE and check_co():
        print(f" ** We have an issue with Open AI using the model {MODEL_INSTRUCT}")
        log_issue(f"No answer despite {max_attempts} attempts", request_chatgpt, "Open AI is down")
    else:
        return rep

def retry_if_too_short(func, *args, **kwargs):
    """
    Retry a given function if its output is too short.
    
    Args:
        func (callable): The function to be called.
        *args: Positional arguments passed to the `func`.
        **kwargs: Keyword arguments passed to the `func`.

        OPTIONAL - you can pass 'min_char_length' and 'max_retries' as parameters.
        min_char_length is the minimum character length to consider the output valid. Defaults to 50.
        max_retries is the minimum the maximum number of times the function should be retried. Defaults to 2.
    
    Returns:
        str: The output of the function if it meets the minimum character length criteria.
        None: If the function output does not meet the criteria after all retries.
    """
    max_retries = kwargs.pop("max_retries", 2)
    min_char_length = kwargs.pop("min_char_length", 50)
    
    for _ in range(max_retries):
        result = func(*args, **kwargs)
        if result and len(result) >= min_char_length:
            return result
    return None

def sanitize_bad_gpt_output(gpt_output: str) -> str:
    """
    Sanitize bad outputs made by GPT according to bad output we already saw.

    Returns:
        - str: The cleaned gpt_output - always strip() the input
    """
    # Check for starting with the assistant prefixes
    if gpt_output.startswith(("Assistant: ", "assistant: ")):
        gpt_output = gpt_output[11:]
    # Check for starting and ending with single or double quotes
    if (gpt_output.startswith("'") and gpt_output.endswith("'")) or (gpt_output.startswith('"') and gpt_output.endswith('"')):
        gpt_output = gpt_output[1:-1]
    return gpt_output.strip()

def self_affirmation_role(role_chatbot_in_text: str) -> str:
    '''
    Function to transform an instruction of the system prompt into a self-affirmation message.

    Theory is that seeing the message twice will make the LLM believe it more.
    '''
    clean_text = role_chatbot_in_text.strip()
    clean_text = clean_text.replace(" you are ", " I am ").replace(" You are ", " I am ").replace(" You Are ", " I Am ")
    clean_text = clean_text.replace("You ", "I ").replace(" you ", " I ").replace(" You ", " I ")
    clean_text = clean_text.replace("Your ", "My ").replace(" your ", " my ").replace(" Your ", " My ")
    return clean_text

# *************************************************************

if __name__ == "__main__":
    pass