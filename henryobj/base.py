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

MAX_TOKEN_CHAT_INSTRUCT = 4097 # Max is 4,097 tokens - this 

MODEL_GPT4 = r"gpt-4"
MODEL_GPT4_LATEST = r"gpt-4-0613"

MAX_TOKEN_GPT4 = 8192 # This is for both the message and the completion

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
        max_tokens (int, optional): Maximum number of tokens to be used in general
        check_length (bool, optional): Will perform an aproximate check on the length of the input not to query if too heavy. For now, we limit to 4K.
    Returns:
        str: The model's reply to the question.

    Note:
        If max_tokens is left to 4000, a print statement will prompt you to adjust it.
    """
    initial_token_usage = calculate_token(role) + calculate_token(question)
    if check_length and initial_token_usage > max_tokens:
        print("Your input is too large for the query. Increase the max_tokens or use 'new_chunk_text' to chunk the text beforehand.")
        return ""
    if max_tokens == 4000:
        print(f"""\nWarning: You are using default maximum response length of about 3000 words.\nIf you don't need that much, it will be faster and cheaper to adjust the max_token.\n
              You are using {initial_token_usage} tokens with the question. For example: if you want an answer of 300 words, put max_token to aprx. {440+initial_token_usage}\n\n
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
    return request_gpt_instruct(instructions=instructions, max_tokens=max_request)

def ask_question_gpt4(role: str, question: str, model=MODEL_GPT4, max_tokens=8000, check_length=True) -> str:
    """
    Queries Chat GPT 4 with a specific question. 

    Args:
        role (str): The system prompt to be initialized in the chat table. How you want ChatGPT to behave.
        question (str): The question to ask the model.
        model (str, optional): The model to use. Defaults to GPT4. Althought it says GPT4, you can use the ChatModel
        max_tokens (int, optional): Maximum number of tokens for the context. Default is 8000 which is huge.
        check_length (bool, optional): Will perform an aproximate check on the length of the input not to query GPT if too long. For now, we limit to 4K only.
    Returns:
        str: The model's reply to the question.

    Note:
        If max_tokens is set to 8000, a print statement will prompt you to adjust it.
    """
    maxi = 8192 if model == MODEL_CHAT else 4097
    initial_token_usage = calculate_token(role) + calculate_token(question)
    if check_length and calculate_token(role) + calculate_token(question) > maxi:
        print("Your input is too large for the query. Increase the max_tokens or use 'new_chunk_text' to chunk the text beforehand.")
        return ""
    current_chat = initialize_role_in_chatTable(role)
    current_chat = add_content_to_chatTable(question, "user", current_chat)
    if max_tokens > 4000 and model == MODEL_CHAT:
        print("The Chat Model has a maximum context of 4097 token. Please adjust the max_token to a value below that.")
        return ""
    if max_tokens == 8000:
        print(f"""\nWarning: You are using default maximum response length of about 3000 words.\nIf you don't need that much, it will be faster and cheaper to adjust the max_token.\n
              You are using {initial_token_usage} tokens with the question. For example: if you want an answer of 300 words, put max_token to aprx. {440+initial_token_usage}\n\n
              """)
        max_request = max_tokens - initial_token_usage
    else:
        max_request = max_tokens
    return request_chatgpt(current_chat, max_token=max_request, model=model)

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

def request_chatgpt(current_chat : list, max_token : int, stop_list = False, max_attempts = 3, model = MODEL_CHAT, temperature = 0, top_p = 1):
    """
    Calls the ChatGPT OpenAI completion endpoint with specified parameters.

    Args:
        current_chat (list): The prompt used for the request.
        max_token (int): The maximum number of tokens to be used in the context (context = reply + question and role)
        stop_list (bool, optional): Whether to use specific stop tokens. Defaults to False.
        max_attempts (int, optional): Maximum number of retries. Defaults to 3.
        model (str, optional): ChatGPT OpenAI model used for the request. Defaults to 'MODEL_CHAT'.
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
        log_issue(f"No answer despite {max_attempts} attempts", request_chatgpt, "Open AI is down")
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
            if 'Rate limit reached' in e:
                print(f"Rate limit reached. We will slow down and sleep for 300ms. This was attempt number {attempts}/{max_attempts}")
                time.sleep(0.3)
            else:
                print(f"Error. This is attempt number {attempts}/{max_attempts}. The exception is {e}. Trying again")
                rep = OPEN_AI_ISSUE
    if rep == OPEN_AI_ISSUE and check_co():
        print(f" ** We have an issue with Open AI using the model {MODEL_INSTRUCT}")
        log_issue(f"No answer despite {max_attempts} attempts", request_chatgpt, "Open AI is down")
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

x = """

"Hi, Henry. Are we waiting for someone else? Let's wait a few minutes. I don't think Ramzi will join because he's overwhelmed. Let's give him a few minutes to join. Sure. So, both of you are based in Lebanon? Yes. Did we meet already? I think you made the presentation for us. Yes, in September or October. What are you both working on or in charge of at TELESEL? I am the head of Enterprise Solutions, Hassan Abdullah. We have been working on chatbot solutions for two years. We have a platform from Cinch. I'm not sure if you are familiar with Cinch. We have the Cinch platform and we are developing chatbots on this platform for our customers. In addition to the contact center provided by Cinch, which is omnichannel as well. It's like a complete solution between the chatbot and the contact center to provide the customer experience to our customers. We have other projects as well. We have different products, not directly related to AI and machine learning. Business solutions like ARP, IoT and other stuff. Hassan is my lead technical engineer. So, basically you are adapting and reselling existing solutions from Cinch and other providers. The platform from Cinch is not a graphical platform. You have to build the flows, the intents, the APIs. It's a kind of development, but a drag and drop development. You write code to do something that is not available by the blocks. Usually it's a kind of integration or you want to do some conversion. That is not in the blocks. You have to have the mindset of a programmer in order to build it properly. I see. Today, and I guess we will continue delving into the topic directly because I'm not sure Ramzi will join. Let's dive in. Today you want to build some of those solutions in-house. Is this the game plan? Actually, we are working on it. As Mohamed Damouche said in the email, we want to explore what you are working on. And see if this can benefit us in what we are working on or maybe in other parts of the business. Voice, maybe on SMS and so on. We are trying to see if there is a benefit from what you are working on. Okay, so you know already what I'm working on or no? From your email that you sent. We tried to do some research on what you have sent. I think Hussain did the research and tried to explore what's in FastAPI. It will be better I think because we didn't have enough time to do it. Maybe you can explain better. Sure. So you're both familiar of course. You're both technical and you're both familiar with OpenAI and ChartsGPT, right? Yes. Sorry Hamdi, can you make a recording for the call? Yeah, of course. Please go ahead. I can do it from my side? Good question. Otherwise, if you just want the transcript, you can record with your phone or you want also the video. If it's possible, please record all of the meeting. Okay, so in this case we'll have to use a screen recorder. There is no option in Google Meet to record? I don't see one. Maybe there is, but I don't see. I'm looking at the same time. I can see change layout, full screen, visual effects, settings, caption. I don't see it from Google Meet. If you don't have a screen recorder, I can record my screen and then show you the video. Is this good for you? Okay, please. One second. Start recording. Alright, so I'm now recording my screen. Would you like me to start with explaining a bit the solutions that I've built? Yeah. Okay, so as I said, you're both familiar with the chat GPT and one of the main objectives is to be able to provide chat GPT-like standards, so the same technology, to your end user, any of your clients, and be able to fine-tune this chat GPT for the knowledge base of your client. Now I'm going to be a bit more explicit. Let's say that one of your clients is an SMB, a small and medium business, travel agency, for example. This travel agency has X destination, 20 destinations. They have certain rules and regulations, certain information about their company. All this information is context. If you ask today chat GPT about this travel agency, chat GPT cannot answer because he doesn't have this context. He is not trained on this data. Now the idea is you want to give this travel agency a chat GPT that knows its data, the data of the travel agency. And basically I did an API that allows to do that with any kind of data input. So you can feed PDF, you can feed text, you can feed a website, and then the data will be kept and used by the chatbot when it gives answer. So you can basically have a custom chat GPT for each client. Now what does it mean that it is into an API? It means that you don't have to use my dashboard or you don't have to use my front end. You can just make requests to the API. And you can make requests both to create the chatbot and to receive the messages. Therefore, you can integrate the solution in any tool that you already have. It's just basic requests. So this is an API request. So POST, GET, and PUT and so on. So your solution does not have a front end? I did a front end for Telegram in particular and I can show it to you and it's very basic, but it works. But the idea, and this is based on the conversation with Mohamed Damouche, is that what you are building is something that is omnichannel. And you want your clients to basically have your solution available on whatever platform they use. Whether it is WhatsApp, Messenger, and so on and so forth. Building the front end and the client and all the integration for that is a different job basically. It's not what I'm doing. And I told him if this is what you want to build, there are many solutions that already do that. But if you want to do it from scratch, even then I would not recommend working with me. I would recommend working with another agency or your own team. I'm only specialized if you want on the custom chatbots. So you want to have chat GPT standard, the engine. You want to customize it. You want the chat to have a memory and context. And you don't want to implement all the code for it. What you mentioned is that you feed the website, you feed the chatbot the API, the website. And it learns directly. This is something that you don't need to access the database, for example, of the travel agency to get this information. You can just get it using, just feeding the website. It will discover and get the information from the website. Everything that will be text on the website can be scrapped, of course. Now, if you want to improve the quality of the answer, it's obviously best to check what knowledge it has learned and improve it or fine tune it. So let's say that you are one of the client benefiting from it. So you Hassan, you have your company and you want a chatbot for customer service for the sake of the discussion and example. We put into it all the information about your stock, your inventory, your return policy, and so on and so forth. Tomorrow you change something. You decide that it's no longer 14 days for the return, but it's 12 days. Well, you will need to update this information in the knowledge base of the chatbot. So, yes, it can feed all the data, it can get all the data, but of course you want to make sure that the context it has is the one that is the latest and the most up to date for your business. So the travel agency shouldn't, I mean, you do whatever you want, but my recommendation is it shouldn't be just take all the website, all the content, and then we don't check what the chatbot will answer. For example, here you can make direct integration, let's say, with the CRM system that they are using. It can take live data directly from the database. You don't need to update it each time. This is actually the question. Are you connecting with the database or you're just connecting the website? Are you getting the information from the website? You can send any information you want. At the end, it's a POST request, and you can update the information. So if we take the example of CRM, either you have in a script that every time there is an update, you send an update to the API and it updates the model. And the update function is done already. It's just like on your side, you have a script that detects because how does the API know that your CRM has a new entry or that the entry has changed? The API is like a server, and you guys know that. So this is the API, and on this side you have the CRM of the client. If you want the data to be updated automatically, you add here, and I can help for that. I'm sure you guys are more competent than me to do it. But of course, the idea is you add a script that says whenever there is a new entry or an update on the CRM, send either a POST or an update request to the API, and the model will be updated automatically. For CRM especially, the CRM has that information about customers usually and all users. So it's constant modification about the data, what's inside. So here you will need a lot of POST in order to keep, because what we usually do in our system is that when we connect to the chatbot, we'll not have the information inside. You will request it when needed from the CRM. This is what we do, because we don't want any... The chatbot is just an interface that will connect you to the information that you need from a CRM, from a database, and from so on. But it itself, it cannot have all this information. Maybe for example, if the different bundles you are selling, the different... This information, yes, it can be updated. And this is also a good way to have it done by API to update the flow. Here we are updating the flow instead of giving him, for example, today we have three bundles, tomorrow we have four bundles. So this data through API is easier and quicker than going into the chatbot flow and changing and updating the box, let's say, that we program. This functionality, it helps with updating the flow of the bot or how it's programmed, but for information inside the CRM, let's say, it's better to keep it by request, otherwise you will have a huge interaction. Look, I think, because here, when you mentioned CRM, it's unclear who are you referring to, because at the end, I understand what you're saying from a theoretical perspective, is that there is data storage, right? And then the chatbot is going to retrieve the relevant data from whatever data storage you have, whatever it is. And this is very clear. Now, what I'm trying to explain is that what I do is that I keep the relevant data into another database to do vector search into it so that the chatbot knows which information is the most relevant. In order to do this vector search, I need to have the data stored on my side, basically, because otherwise it would take too much time. Imagine if the data is stored on your side, and I'm doing the request every time someone is chatting with the chatbot. It means that we would have to vectorize the data on your side every time, then perform a co-scene similarity, and then retrieve. You know what I mean? Because of how you programmed it, that's fine. That's fine. This is the way you are doing it. It will require this thing. It can be done differently, but the rule today with the new LLMs model, large language model like OpenAI, is to use vector-based search. So, each piece of data is a chunk of content, is transformed into a vector, it's called embeddings. And this is what they use to provide the model the right context. Because imagine your database has one million entries, and there are only three entries that are relevant for a specific query. So, to pass through all this as quickly as possible, they use this vector-based search, co-scene similarity, basically, to see how much the question and the data are similar in terms of embeddings. How are you building it? Is it built by intent? Or does it use the chatGPT model, plus it adds on top of it this vectorization that you are doing for the client? Yes. So, it depends on which part we are talking. You have the memory of the conversation that is in-house. You have the embedding that is using the ADA002 model, which is also the model used by OpenAI. And then there is the engine for speaking that is also an API call to OpenAI. And this is the interesting part, you can change this engine whenever you want. So, tomorrow you have a better engine or a cheaper engine. It's just like, which engine are we calling? This is the question and answer. The vectorization, the embedding is done with another model called ADA. And then the memory is something that I built myself. So, it's combining the three, basically. And to come back to what you were saying earlier for the CRM, I think anything, and you guys know it better than me, anything technically is possible. It depends on what do we want to prioritize. Do we want to prioritize speed? And in this case, you don't want to do the same operation over and over. You want to have vectorization done once. Do we want to prioritize privacy or security? In this case, we keep the data on your side and we just do the request that we need. Do we want to prioritize a user experience? And therefore, it should be as quickly and simple as possible for your clients. So, it all depends on your objectives at the end. Okay. What we are providing to businesses is like a specific chatbot. We are not providing a chatbot, although the system that we have, the platform has an integration shared GPT. So, we are leveraging on that as well. But for a specific business, they have specific needs. For example, if I am a mobile operator, I need to check my balance. For example, I'm building a flow who will request information from the database of the mobile operator to get the balance. I can recharge, I can credit transfer. So, this is basic functionalities that our chatbot is doing. In addition to being more informational, which is asking questions and conversational, asking questions and replying. This is where I think I need to understand from you how you can benefit us specific for a bank. For example, if I'm doing transaction on a bank, I'm doing identification, identity verification, for example, before I'm doing the transaction, I'm checking my balance. So, this is all transaction-based functionalities that we are doing for our businesses. So, I can get it from the conversational one, which should be informational. I can give him whatever bundles I have, I can give him whatever services I can provide. You can do it based on what you mentioned. But for the transaction here, how can this be done using your model? Yes, excellent question, Hassan. I think this is the smartest way to approach the problem, is to distinguish, and you call it transactional. Let's use this terminology, is to distinguish between the conversation and the operations, the transactions. The transaction has to be defined, and it's already defined by you and by the client, because it has to be very specific. You don't want the model to interpret and decide on its own what to do. So, what you're doing shouldn't change on this aspect. It's perfect. This model is for the conversation, the natural conversation, where the user or the visitor can ask more general questions and something that resembles more the support or the assistance that a human would provide. This is why we're using, of course, a chat GPT. Now, how does this model would be useful to you? Is that your client, the bank, let's continue with the example of the bank. The bank has, I don't know, maybe 100 pages or 50 pages of information about the bank. If the visitor, I'm the visitor, he is requesting to check his account, his balance, it goes into the pipeline transaction. It doesn't call my API. If he is asking, like, when are you opening, or at what time is it closing, or is this person working there, whatever is possibly in the 50 pages of information about the bank, then you do an API call, and it returns a normal conversation as if it's a human agent answering to him. So this is how the model would work. Can you, as you mentioned, you have specific, something developed for each client, you will have to develop a separate instance. Yes? Yes, in the database, but this, it's you, you can create your client an API call automatically. So it's like a post request, basically. I'm trying to see if I can use it while I'm using, because for transactional and operational, I need to do the integration that I'm doing, using the Cinch chatbot. So I'm trying to see if for those conversational requests, I can use your solution, for example. So, but here also, each, any of the chatbot that will be the API that a bank will use, it's different than the API that a, let's say mobile operator will use, and CRM will use"""

Role = "You are a strategy consultant, you need to summarize the call while identifying who said what. For that, first identify the speakers, then link the content to each speaker. Only when you did this, you can proceed with a summary. Return the text with bullet points for each speaker."
if __name__ == "__main__":
    print(ask_question_gpt4(Role, x + x))
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