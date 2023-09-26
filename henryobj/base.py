"""
    @Author:				Henry Obegi <HenryObj>
    @Email:					hobegi@gmail.com
    @Creation:				Friday 1st of September
    @LastModif:             Saterday 2nd of September
    @Filename:				base.py
    @Purpose                All the utility functions
    @Partof                 Spar
"""

# ************** IMPORTS ****************
import openai
import os
import requests
import datetime
import inspect
import tiktoken
import json
from typing import Callable, Any, Union, List, Dict
import re
import time
from urllib.parse import urlparse, urlunparse, quote, unquote
import random

# ****** PATHS & GLOBAL VARIABLES *******

OAI_KEY = os.getenv("OAI_API_KEY")
openai.api_key = OAI_KEY

MODEL_CHAT = r"gpt-3.5-turbo"
MODEL_INSTRUCT = r"gpt-3.5-turbo-instruct"
MODEL_CHAT_LATEST = r"gpt-3.5-turbo-0613"

MAX_TOKEN_CHAT = 4097 # Max is 4,097 tokens

MODEL_GPT4 = r"gpt-4"
MODEL_GPT4_LATEST = r"gpt-4-0613"

MAX_TOKEN_GPT4 = 8192 # For some reason (03/09/23) an input above 4097 failed so I guess this is for the response? 

OPEN_AI_ISSUE = r"%$144$%" # When OpenAI is down
MODEL_EMB = r"text-embedding-ada-002"


# *************************************************************************************************
# *************************************** General Utilities ***************************************
# *************************************************************************************************

def correct_spaces_in_text(text):
    '''
    Ensures punctuation marks like ".", "?", "!", ";", and "," are correctly spaced with only one space with the next non-space char. 

    Example: 'Hello,world!How     are you?' would be converted to 'Hello, world! How are you?'
    '''
    return re.sub(r"([\.,\?!;])\s*(\S)", r"\1 \2", text)

def is_json(myjson: str) -> bool:
  '''
  Returns True if the input is in json format. False otherwise.
  '''
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

def log_issue(exception, function, message):
    # Your log_issue implementation would be here.
    # For this example, I'm using a simple print statement.
    print(f"Error in function {function.__name__}: {exception}. Message: {message}")

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
    print(f" * ERROR HO144 * Issue in module {module_name} with {func.__name__} ** Info: {additional_info} ** Exception: {exception} ** When: {now}\n")

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
    New function to chunk the text in better blocks.
    The idea is to pass several times and make the ideal blocks first (rather than one time targetting the ideal token) then breaking the long ones.
    target_token will be used to make chunks that get close to this size. Returns the chunk_oai if issue.
    '''
    def find_sentence_boundary(chunk, end, buffer_char):
        for punct in ('. ', '.', '!', ';'):
            pos = chunk[:end].rfind(punct)
            if pos != -1 and end - pos < buffer_char:
                return pos + len(punct), "best"
        return end, "worst"
    if calculate_token_aproximatively(text) < 1.1 * target_token:
        return [text]
    try:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        buffer_char = 40 * 4
        merged_chunks = []
        i = 0
        while i < len(paragraphs):
            current_token_count = calculate_token_aproximatively(paragraphs[i])
            if current_token_count < target_token * 0.5:
                if i == 0:
                    merged_chunks.append(paragraphs[0] + ' ' + paragraphs[1])
                    i = 2
                elif i == len(paragraphs) - 1:
                    merged_chunks[-1] += ' ' + paragraphs[i]
                    break
                else:
                    if calculate_token_aproximatively(paragraphs[i-1]) < calculate_token_aproximatively(paragraphs[i+1]):
                        merged_chunks[-1] += ' ' + paragraphs[i]
                        i += 1
                    else:
                        merged_chunks.append(paragraphs[i] + ' ' + paragraphs[i+1])
                        i += 2
            else:
                merged_chunks.append(paragraphs[i])
                i += 1

        final_chunks = []
        for chunk in merged_chunks:
            chunk_token_count = calculate_token_aproximatively(chunk)
            if chunk_token_count > target_token * 1.5:
                end = target_token * 4
                remaining_tokens = chunk_token_count
                while remaining_tokens > target_token:
                    cut_pos, grade = find_sentence_boundary(chunk, end, buffer_char)
                    final_chunks.append(chunk[:cut_pos])
                    chunk = chunk[cut_pos:]
                    remaining_tokens = calculate_token_aproximatively(chunk)
                if chunk:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        return final_chunks
    except Exception as e:
        log_issue(e, new_chunk_text)
        return ""
        # We could have a back up function here

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

# *************************************************************************************************
# ****************************************** GPT Related ******************************************
# *************************************************************************************************

def add_content_to_chatTable(content: str, role: str, chatTable: List[Dict[str, str]]) -> List[Dict[str, str]]:
    '''
    Feeds a chatTable with the new query. Returns the new chatTable.
    Role is either 'assistant' when the AI is answering or 'user' when the user has a question.
    Added a security in case change of name.
    '''
    if role in ["user", "assistant"]:
        chatTable.append({"role":f"{role}", "content": f"{content}"})
        return chatTable
    else:
        #log_issue("Wrong entry for the chattable", add_content_to_chatTable, f"For the role {role}")
        if role in ["User", "Client", "client"]:
            chatTable.append({"role":"user", "content": f"{content}"})
        else:
            chatTable.append({"role":"assistant", "content": f"{content}"})
        return chatTable

def ask_question_gpt(question: str, role ="", max_tokens=4000, check_length=True) -> str:
    """
    Queries OpenAI Instruct Model with a specific question. This has a better performance than getting a chat completion.

    Args:
        question (str): The question to ask the model.
        role (str, optional): As the legacy method would initialize a role, you can use previously defined role which will be part of the prompt.
        max_tokens (int, optional): Maximum number of tokens for the reply.
        check_length (bool, optional): Will perform an aproximate check on the length of the input not to query if too heavy. For now, we limit to 4K.
    Returns:
        str: The model's reply to the question.

    Note:
        If max_tokens is set to 4000, a print statement will prompt you to adjust it.
    """
    if check_length and calculate_token_aproximatively(role) + calculate_token_aproximatively(question) > 4000:
        print("Your input is likely too long for one query. You can use 'new_chunk_text' to chunk the text beforehand.")
        return ""
    if max_tokens == 4000:
        print(f"\nWarning: You are using default maximum response length of about 3000 words.\nIf you don't need that much, it will be faster and cheaper to adjust the max_token.\n\n")
    if role == "": instructions = question
    else:
        instructions = f"""You must follow strictly the Role to answer the Question.
        \nRole = {role}
        \n
        Question = {question}
        \nMake sure you take your time to understand the Role and follow the Role before answering the Question. Important: Answer ONLY the Question and nothing else.
        """
    return request_gpt_instruct(instructions=instructions, max_tokens=max_tokens)

def ask_question_gpt4(role: str, question: str, model=MODEL_GPT4, max_tokens=4000, check_length=True) -> str:
    """
    Queries Chat GPT 4 with a specific question.

    Args:
        role (str): The system prompt to be initialized in the chat table. How you want ChatGPT to behave.
        question (str): The question to ask the model.
        model (str, optional): The model to use. Defaults to GPT4.
        max_tokens (int, optional): Maximum number of tokens for the reply. Defaults to 4000.
        check_length (bool, optional): Will perform an aproximate check on the length of the input not to query GPT if too long. For now, we limit to 4K only.
    Returns:
        str: The model's reply to the question.

    Note:
        If max_tokens is set to 4000, a print statement will prompt you to adjust it.
    """
    if model in [MODEL_CHAT, MODEL_GPT4]: maxi = 4000
    if check_length and calculate_token_aproximatively(role) + calculate_token_aproximatively(question) > maxi:
        print("Your input is likely too long for one query. You can use 'new_chunk_text' to chunk the text beforehand.")
        return ""
    current_chat = initialize_role_in_chatTable(role)
    current_chat = add_content_to_chatTable(question, "user", current_chat)
    if max_tokens == 4000:
        print(f"\nWarning: You are querying {model} with the default response length of about 3000 words.\nIf you don't need that much, it will be faster and cheaper to adjust the max_token.\n\n")
    return request_gpt(current_chat, max_token=max_tokens, model=model)

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
        log_issue("Previous_chat is of 0 len", change_role_chatTable)
        return [{'role': 'system', 'content': new_role}]
    previous_chat.pop(0)
    return [{'role': 'system', 'content': new_role}] + previous_chat

def embed_text(text: str, max_attempts: int = 3):
    '''
    Micro function which returns the embedding of one chunk of text or 0 if issue.
    Used for the multi-threading.
    '''
    res = 0
    if text == "": return res
    attempts = 0
    while attempts < max_attempts:
        try:
            res = openai.Embedding.create(input=text, engine=MODEL_EMB)['data'][0]['embedding']
            return res
        except Exception as e:
            attempts += 1
    if check_co(): log_issue(f"No answer despite {max_attempts} attempts", embed_text, "Open AI is down")
    return res

def initialize_role_in_chatTable(role_definition: str) -> List[Dict[str, str]]:
    '''
    We need to define how we want our model to perform.
    This function takes this definition as a input and returns it into the chat_table_format.
    '''
    return [{"role":"system", "content":role_definition}]

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

def request_gpt(current_chat : list, max_token : int, stop_list = False, max_attempts = 3, model = MODEL_CHAT, temperature = 0, top_p = 1):
    """
    Calls the OpenAI completion endpoint with specified parameters.

    Args:
        current_chat (list): The prompt used for the request.
        max_token (int): The maximum number of tokens in the reply.
        stop_list (bool, optional): Whether to use specific stop tokens. Defaults to False.
        max_attempts (int, optional): Maximum number of retries. Defaults to 3.
        model (str, optional): OpenAI model used for the request. Defaults to 'MODEL_CHAT'.
        temperature (float, optional): Sampling temperature for the response. A value of 0 means deterministic output. Defaults to 0.
        top_p (float, optional): Nucleus sampling parameter, with 1 being 'take the best'. Defaults to 1.

    Returns:
        str: The response text or 'OPEN_AI_ISSUE' if an error occurs (e.g., if OpenAI service is down).
    """
    stop = stop_list if (stop_list and len(stop_list) < 4) else ""
    attempts = 0
    valid = False
    #print("Writing the reply for ", current_chat) # Remove in production - to see what is actually fed as a prompt
    while attempts < max_attempts and not valid:
        try:
            response = openai.ChatCompletion.create(
                messages= current_chat,
                temperature=temperature,
                max_tokens= int(max_token),
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
                model= model,
            )
            rep = response["choices"][0]["message"]["content"]
            rep = rep.strip()
            valid = True
        except Exception as e:
            attempts += 1
            if 'Rate limit reached' in e:
                print(f"Rate limit reached. We will slow down and sleep for 300ms. This was attempt number {attempts}/{max_attempts}")
                time.sleep(0.3)
            else:
                print(f"Error. This is attempt number {attempts}/{max_attempts}. The exception is {e}. Trying again")
                rep = OPEN_AI_ISSUE
    if rep == OPEN_AI_ISSUE and check_co():
        print(f" ** We have an issue with Open AI using the model {model}")
        log_issue(f"No answer despite {max_attempts} attempts", request_gpt, "Open AI is down")
    return rep

def request_gpt_instruct(instructions : str, max_tokens = 300, max_attempts = 3, temperature = 0, top_p = 1) -> str:
    '''
    Calls the OpenAI completion endpoint with specified parameters.

    Args:
        instructions (str): The prompt used for the request.
        max_token (int): The maximum number of tokens in the reply - defaulted to 300 (200 words)
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
                    temperature = 0,
                    max_tokens = max_tokens,
                    top_p =1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            rep = rep["choices"][0]["text"].strip()
            valid = True
        except Exception as e:
            attempts += 1
            if 'Rate limit reached' in e:
                print(f"Rate limit reached. We will slow down and sleep for 300ms. This was attempt number {attempts}/{max_attempts}")
                time.sleep(0.3)
            else:
                print(f"Error. This is attempt number {attempts}/{max_attempts}. The exception is {e}. Trying again")
                rep = OPEN_AI_ISSUE
    if rep == OPEN_AI_ISSUE and check_co():
        print(f" ** We have an issue with Open AI using the model {MODEL_INSTRUCT}")
        log_issue(f"No answer despite {max_attempts} attempts", request_gpt, "Open AI is down")
    return rep

def print_gpt_models():
    '''
    To list the gpt models provided by OpenAI.
    '''
    response = openai.Model.list() # list all models

    for elem in response["data"]:
        name = elem["id"]
        if "gpt" in name or "embedding" in name: print(name)

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
    #print(ask_question_gpt("What is the meaning of life? Give a long answer with references."))
    pass

# Testing chat completion vs instruct:
# Huge improvement (7sc vs 1sc) for instruct on the question side
"""
    start = time.time()
    y = ask_question_gpt("You are an intelligent assistant", "What is the meaning of life", max_tokens=300)
    one = time.time()
    x = request_gpt_instruct("What is the meaning of life?")
    end = time.time()
    print(f"One is {one - start} sc // Two is {end - one}")
    print(f"Chat {y} \n\n")
    print("Instruct", x)
"""

# Testing tiktoken vs aproximation
"""
with open("/Users/henry/Coding/mypip/base/longtext.txt", "r") as file:
        w = file.read()
    
    start = time.time()
    for i in range(1000):
       t = calculate_token_aproximatively(w)
    
    t1 = time.time()
    for i in range(1000):
        hh = calculate_token(w)

    end = time.time()
    print(f"First was done in {t1 - start} seconds - value is {t}. Second was done in {end - t1} seconds value is {hh}")
"""