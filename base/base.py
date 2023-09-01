"""
    @Author:				Henry Obegi <HenryObj>
    @Email:					hobegi@gmail.com
    @Creation:				Friday 1st of September
    @LastModif:             
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
from typing import Callable, Any
import re
import time

# ****** PATHS & GLOBAL VARIABLES *******

OAI_KEY = os.getenv("OAI_API_KEY")
openai.api_key = OAI_KEY

MODEL_CHAT = r"gpt-3.5-turbo"
MODEL_GPT4 = r"gpt-4"
OPEN_AI_ISSUE = r"%$144$%" # When OpenAI is down
MODEL_EMB = r"text-embedding-ada-002"


# ****************** FUNCS ******************


# *************************************************************************************************
# *************************************** General Utilities ***************************************
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

def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

def get_content_of_file(file_path : str) -> str:
    '''
    Mini function not to write everytime the with open etc.
    '''
    with open(file_path,"r") as file:
        x = file.read()
    return x

def get_module_name(func) -> str:
    '''
    Given a function, returns the name of the module in which it is defined.
    '''
    module = inspect.getmodule(func)
    if module is None:
        return ''
    else:
        return module.__name__.split('.')[-1]

def get_now(time = False) -> str:
    '''
    Small function to get the timestamp in string format.
    By default we return the following format: "10_Jan_2023" but if time is True, we will return 10_Jan_2023_@15h23s33
    '''
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, "%d_%b_%Y@%Hh%Ms%S") if time else datetime.datetime.strftime(now, "%d_%b_%Y")

def log_issue(exception: Exception, func: Callable[..., Any], additional_info: str = "") -> None:
    '''
    Logs an issue. Can be called anywhere and will display an error message showing the module, the function, the exception and if specified, the additional info.

    Parameters:
    - exception (Exception): The exception that was raised.
    - func (Callable[..., Any]): The function in which the exception occurred.
    - additional_info (str): Any additional information to log. Default is an empty string.

    Returns:
    None
    '''
    now = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
    module_name = get_module_name(func)
    print(f" * ERROR HO144 * Issue in module {module_name} with {func.__name__} ** Info: {additional_info} ** Exception: {exception} ** When: {now}\n")

# local tests
def lprint(*args):
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

def new_chunk_text(text: str, target_token = 200) -> list:
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

def perf(function):
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

def remove_break_lines(text):
    '''
    Removes all double space and '\n'.
    Returns the text.
    '''
    jump = '\n'
    double_space = '  '
    while jump in text:
        text = text.replace(jump, ' ')
    while double_space in text:
        text = text.replace(double_space, ' ')
    return text

def remove_jump_double_punc(text):
    '''
    Removes all '\n' and '..' for the function to analyze sentiments.
    '''
    jump = '\n'
    text = text.replace(jump,'')
    double = '..'
    while double in text:
        text = text.replace(double,'.')
    return text

def remove_excess(text :str) -> str:
    '''
    Removes all double space and doubles break lines '\n\n'.
    Returns the text.
    '''
    double_jump = '\n\n'
    double_space = '  '
    while double_jump in text:
        text = text.replace(double_jump, '\n')
    while double_space in text:
        text = text.replace(double_space, ' ')
    return text

def sanitize_json_response(response):
    '''
    Function to make sure we have a json-like structure.
    Returns False if issue, otherwise returns the sanitized answer.
    '''
    (bal1,bal2) = response.find("{"), response.find("}")
    if bal1 < 0 or bal2 < 0: return False
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

def add_content_to_chatTable(content : str, role : str, chatTable : list) -> list:
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

def ask_question_gpt(role : str, question : str, model = MODEL_CHAT, max_tokens = 4000) -> str:
    '''
    Function to ask a question to ChatGPT. Returns the reply to this question in a text format.
    You must define a role.
    '''
    current_chat =  initialize_role_in_chatTable(role) # we create the chatTable
    current_chat = add_content_to_chatTable(question, "user", current_chat) # we add the question
    if max_tokens == 4000:
        print(f"Will ask GPT and the answer is max: {max_tokens} for the question ****STARTQUESTION*****{question}\n****ENDQUESTION*****")
        #max_tokens -= (calculate_token_aproximatively(role)+ calculate_token_aproximatively(question))
    #print(f"Will ask GPT and the answer is max: {max_tokens} for the question ****STARTQUESTION*****{question}\n****ENDQUESTION*****")
    return request_gpt(current_chat, max_token=max_tokens, model=model)

# Calculating token is too expensive in time (about 1 sec for 1000 words)
def calculate_token(text : str) -> int:
    '''
    New version with the tokenizer api. Takes about 0.13 seconds.
    No check on whether text is a string so make sure param is correct (force str() before using).
    '''
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def calculate_token_aproximatively(text : str) -> str:
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

def change_role_chatTable(previous_chat : list, new_role : str) -> list:
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

def embed_text(text, max_attempts = 3):
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

def initialize_role_in_chatTable(role_definition : str) -> list:
    '''
    We need to define how we want our model to perform.
    This function takes this definition as a input and returns it into the chat_table_format.
    '''
    return [{"role":"system", "content":role_definition}]

def request_gpt(current_chat, max_token, stop_list = False, max_attempts = 3, model = MODEL_CHAT):
    '''
    Function calling the openAI completion endpoint.
    Parameters are the prompt, the max_token in the reply, the stops we use, and the number of attempt.
    Temperature is at 0 (no 'creativity') and top_p is at 1 ('take the best').
    Returns the text or OPEN_AI_ISSUE is OpenAI is down.
    '''
    stop = stop_list if (stop_list and len(stop_list) < 4) else ""
    attempts = 0
    valid = False
    #print("Writing the reply for ", current_chat) # Remove in production - to see what is actually fed as a prompt
    while attempts < max_attempts and not valid:
        try:
            response = openai.ChatCompletion.create(
                messages= current_chat,
                temperature=0,
                max_tokens= int(max_token),
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
                model= model,
            )
            rep = response["choices"][0]["message"]["content"]
            rep = rep.strip()
            valid = True
        except Exception as e:
            print("Issue OAI ",e)
            attempts += 1
            rep = OPEN_AI_ISSUE
    if rep == OPEN_AI_ISSUE and check_co():
        print("WE HAVE AN ISSUE")
        log_issue(f"No answer despite {max_attempts} attempts", request_gpt, "Open AI is down")
    return rep

def print_gpt_models():
    '''
    We need to check from time to time if there is a new gpt model
    '''
    response = openai.Model.list() # list all models

    for elem in response["data"]:
        name = elem["id"]
        if "gpt" in name or "embedding" in name: print(name)

def self_affirmation_role(role_chatbot_in_text):
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