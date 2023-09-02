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
MODEL_GPT4 = r"gpt-4"
OPEN_AI_ISSUE = r"%$144$%" # When OpenAI is down
MODEL_EMB = r"text-embedding-ada-002"


# *************************************************************************************************
# *************************************** General Utilities ***************************************
# *************************************************************************************************

# Better use tokenizer for production but good enough in the meantime
def add_space_to_punctuation(text):
    '''
    To ensure that any "." "?" "!" ";" and "," is followed by a space and doesn't have a before space.
    '''
    return re.sub(r"([\.,\?!;])\s*(\S)", r"\1 \2", text)

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

def is_json(myjson: str) -> bool:
  '''
  Returns True if the input is in json format. False otherwise.
  '''
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

def format_datetime(datetime):
    '''
    Takes a Datetime as an input and returns a string in the format "10-Jan-2022"
    '''
    return datetime.strftime('%d-%b-%Y')

def generate_unique_integer():
    '''
    Returns a random integer. Should be unique because between 0 and 2*32 -1 but still we can check after.
    '''
    # Generate a random number within the range of a 32-bit integer
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

def get_now(exact: bool = False) -> str:
    '''
    Small function to get the timestamp in string format.
    By default we return the following format: "10_Jan_2023" but if exact is True, we will return 10_Jan_2023_@15h23s33
    '''
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, "%d_%b_%Y@%Hh%Ms%S") if exact else datetime.datetime.strftime(now, "%d_%b_%Y")

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
    Strong cleaner which removes any char that is not ascii.
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

def ask_question_gpt(role: str, question: str, model=MODEL_CHAT, max_tokens=4000, check_length=True) -> str:
    """
    Queries ChatGPT with a specific question.

    Args:
        role (str): The system prompt to be initialized in the chat table. How you want ChatGPT to behave.
        question (str): The question to ask the model.
        model (str, optional): The model to use. Defaults to the chat model 3.5 turbo.
        max_tokens (int, optional): Maximum number of tokens for the reply. Defaults to 4000.
        check_length (bool, optional): Will perform an aproximate check on the length of the input not to query GPT if too long.
    Returns:
        str: The model's reply to the question.

    Note:
        If max_tokens is set to 4000, a print statement will prompt you to adjust it.
    """
    if check_length and calculate_token_aproximatively(role) + calculate_token_aproximatively(question) > 5000:
        print("Your input is likely too long for one query. You can use 'new_chunk_text' for that")
        return ""
    current_chat = initialize_role_in_chatTable(role)
    current_chat = add_content_to_chatTable(question, "user", current_chat)
    if max_tokens == 4000:
        print(f"You are querying GPT with a maximum response length of about 3000 words for the question: {question}")
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
            print(f"Error. This is attempt number {attempts}/{max_attempts}. The exception is {e}. Trying again")
            rep = OPEN_AI_ISSUE
    if rep == OPEN_AI_ISSUE and check_co():
        print("WE HAVE AN ISSUE")
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