# All that is related ot Open AI

from .config import (
    ERROR_MESSAGE, OPEN_AI_ISSUE, MAX_TOKEN_OUTPUT_DEFAULT, MAX_TOKEN_OUTPUT_DEFAULT_HUGE, MAX_TOKEN_WINDOW_GPT4, 
    MAX_TOKEN_WINDOW_GPT4_TURBO, MAX_TOKEN_WINDOW_OLD, MAX_TOKEN_WINDOW_GPT35_TURBO, MODEL_GPT4_TURBO, MODEL_GPT4O,
    MODEL_GPT4_STABLE, MODEL_CHAT, MODEL_EMB_LARGE, MODEL_CHAT_STABLE, MODEL_CHAT_BACKUP, WINDOW_BUFFER
)
from .base import log_warning, log_issue, split_into_sentences, custom_round, check_co


from typing import Optional

import tiktoken
import openai
import time
import json
import os
import re


# ****************************************** INIT CLIENT *****************************************



OAI_KEY = os.getenv("OAI_API_KEY")
client = openai.OpenAI(
    api_key=OAI_KEY,
)


# ****************************************** SUPPORT TO LLM ***************************************

def add_content_to_chatTable(content: str, role: str, chatTable: list[dict[str, str]]) -> Optional[list[dict[str, str]]]:
    """
    Feeds a chatTable with the new query. Returns the new chatTable.
    Role is either 'assistant' when the AI is answering or 'user' when the user has a question. Returns None if issue

    Note:
        - Security for the content (json_safe)
    """
    new_chatTable = list(chatTable)
    normalized_role = role.lower()
    if normalized_role not in ["user", "assistant"]: 
        log_issue("Wrong role for the Chattable", add_content_to_chatTable, f"Role use is {role}")
        return
    content = make_string_json_safe(content)
    if normalized_role == "user":
        new_chatTable.append({"role": "user", "content": content})
    else:
        new_chatTable.append({"role": "assistant", "content": content})
    return new_chatTable

def calculate_token(text: str) -> Optional[int]:
    """
    Calculates the number of tokens for a given text using a specific tokenizer.

    Args:
        text (str): The text to calculate tokens for.

    Returns:
        int: The number of tokens in the text or -1 if there's an error.
    
    Note:
        Uses the tokenizer API and takes approximately 0.13 seconds per query.
    """
    if not isinstance(text, str): 
        log_warning(f"Input is {type(text)} - must be str. Try force conversation", calculate_token, text)
        try:
            text = str(text)
        except Exception as e:
            log_issue(e, calculate_token, f"Failed to convert to string => {text}")
            return
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except Exception as e:
        log_issue(e, calculate_token, f"Input type: {type(text)}. Text: {text}")
        return -1

def calculate_token_aproximatively(text: str) -> Optional[int]:
    """
    Returns the token cost for a given text input without calling tiktoken.

    2 * Faster than tiktoken but less precise. Will go on the safe side (so real tokens is less)

    Method: A token is about 4 char when it's text but when the char is special, it consumes more token.
    """
    if not isinstance(text, str): 
        log_warning(f"Input is {type(text)} - must be str. Try force conversation", calculate_token, text)
        try:
            text = str(text)
        except Exception as e:
            log_issue(e, calculate_token, f"Failed to convert to string => {text}")
            return
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

def change_role_chatTable(previous_chat: list[dict[str, str]], new_role: str) -> list[dict[str, str]]:
    """
    Function to change the role defined at the beginning of a chat with a new role.
    Returns the new chatTable with the system role updated.
    """
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

def check_if_gptconv_format(conversation: list[dict[str, str]]) -> Optional[bool]:
    """
    Checks if the structure of the list matches the GPT conversation format.
    
    Returns:
    - bool: True if valid, False otherwise.
    """
    allowed_roles = {'user', 'assistant', 'system'}
    for entry in conversation:
        if not set(entry.keys()) == {'role', 'content'}: return
        if not isinstance(entry['role'], str) or not isinstance(entry['content'], str): return
        if entry['role'] not in allowed_roles: return
    return True

def check_valid_gpt_conversation(possible_gpt_conv) -> Optional[bool]:
    """
    Returns True / False depending on whether it is a valid GPT conv.
    
    Args:
        Can be a string or list, it doesn't matter. It will check that the internal elements match the GPT format.
    """
    try:
        gpt_conv = possible_gpt_conv if isinstance(possible_gpt_conv, list) else json.loads(possible_gpt_conv)
        if not isinstance(gpt_conv, list): return
        if not all([isinstance(elem, dict) for elem in gpt_conv]): return
        return check_if_gptconv_format(gpt_conv)
    except:
        return

def new_chunk_text(text: str, target_token: int = 200) -> list[str]:
    """
    Much simpler function to chunk the text in blocks by spliting by sentence. The last chunk might be small.
    """
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
    print(f"We got and returned {len(final_chunks)} chunks")
    return final_chunks

def get_gptconv_readable_format(gpt_conversation: str, system_message: bool = True) -> str:
    """
    Formats a string format GPT conversation (after being extracted from DB) in a human-friendly way.
    If you don't want the system message, mention False. Default, we put it

    Returns:
        - str: The formatted GPT conversation or Error message
    """
    # guard clause
    if not gpt_conversation or not isinstance(gpt_conversation, str): return "Failed to convert to a GPT conversation (not a valid string input)\n"
    def detect_quote_type(s: str) -> str:
        if "\"role\": \"system\", \"content\":" in s:
            return "\""
        elif "'role': 'system', 'content':" in s:
            return "'"
        else:
            return None
    try:
        quote_type = detect_quote_type(gpt_conversation)
        role_user = f"{quote_type}role{quote_type}: {quote_type}user{quote_type}, {quote_type}content{quote_type}:"
        role_assistant = f"{quote_type}role{quote_type}: {quote_type}assistant{quote_type}, {quote_type}content{quote_type}:"
        role_system = f"{quote_type}role{quote_type}: {quote_type}system{quote_type}, {quote_type}content{quote_type}:"
        gpt_conv_as_list = []
        # first, we add the system message
        if system_message:
            anchor_1 = gpt_conversation.find(role_system) # len 28 so 30 with the space
            if anchor_1 == -1:
                log_issue("Failed to convert to a GPT conversation", get_gptconv_readable_format, f"No system prompt - wrong format for the input {gpt_conversation}")
                return ERROR_MESSAGE
            anchor_2 = gpt_conversation.find(role_user) # len 26 so 28 with the space
            anchor_2_alt = gpt_conversation.find(role_assistant) # len 31 so 33 with the space
            # only the system message
            if anchor_2 == -1: 
                print("Warning: Your GPT conversation only contains the system prompt")
                return f"system: {gpt_conversation[anchor_1+30:-3].strip()}"
            elif 0 < anchor_2 < anchor_2_alt or (0 < anchor_2 and anchor_2_alt == -1):
                gpt_conv_as_list.append(['system', gpt_conversation[anchor_1+30:anchor_2-5].strip()]) # -5 comes from "}, {" of the next element
            # Self-affirmation role
            elif 0 < anchor_2_alt < anchor_2:
                gpt_conv_as_list.append(['system', gpt_conversation[anchor_1+30:anchor_2_alt-5].strip()])
            else:
                log_issue("Failed to convert to a GPT conversation", get_gptconv_readable_format, f"Weird structure for the input {gpt_conversation}")
                return ERROR_MESSAGE
        while True:
            anchor_1 = gpt_conversation.find(role_user) # len 26 so 28 with the space
            anchor_2 = gpt_conversation.find(role_assistant) # len 31 so 33 with the space
            if anchor_1 == -1:
                if anchor_2 == -1:
                    break
                gpt_conv_as_list.append(['assistant', gpt_conversation[anchor_2+33:-3].strip()])
                break
            elif anchor_2 == -1:
                gpt_conv_as_list.append(['user', gpt_conversation[anchor_1+28:-3].strip()])
                break
            else:
                if anchor_1 < anchor_2:
                    # safety because sometimes we have several users in a row
                    anchor_1_safety = gpt_conversation[anchor_1+29:].find(role_user)
                    anchor_2 = anchor_2 if anchor_2-anchor_1 < anchor_1_safety or anchor_1_safety == -1 else anchor_1_safety +34 # 29 + 5. 29 comes from searching the string after anchor_1 + 29. The 5 comes from the difference between anchor_2 (33) and anchor_1 (28)
                    gpt_conv_as_list.append(['user', gpt_conversation[anchor_1+28:anchor_2-5].strip()]) # -5 comes from "}, {"
                    gpt_conversation = gpt_conversation[anchor_2-5:]
                else:
                    # safety because we could have several assistants in a row
                    anchor_2_safety = gpt_conversation[anchor_2+34:].find(role_assistant)
                    anchor_1 = anchor_1 if anchor_1-anchor_2 < anchor_2_safety or anchor_2_safety == -1 else anchor_2_safety + 29 # 34 comes from searching the string after anchor_2 + 34 minus the difference between anchor_1 and anchor 2.
                    gpt_conv_as_list.append(['assistant', gpt_conversation[anchor_2+33:anchor_1-5].strip()])
                    gpt_conversation = gpt_conversation[anchor_1-5:]
        result_string = '\n'.join(': '.join(pair) for pair in gpt_conv_as_list)
        return result_string
    except Exception as e:
        log_issue(e, get_gptconv_readable_format, f"This was the input {gpt_conversation}")
        return ERROR_MESSAGE

def initialize_role_in_chatTable(role_definition: str) -> list[dict[str, str]]:
    """
    We need to define how we want our model to perform.
    This function takes this definition as a input and returns it into the chat_table_format.

    Note:
        Makes the role json_safe
    """
    safe_role = make_string_json_safe(role_definition)
    return [{"role":"system", "content":safe_role}]

def make_string_json_safe(s : str) -> str:
    """
    Replace newlines, tabs, and other control characters
    """
    s = s.replace('"', "'")
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    s = s.strip()
    return s

def print_gpt_models(all:bool=True) -> None:
    """
    To list the gpt models provided by OpenAI.
    
    Args:
        all: If True, will print all the models. Else, only the 'GPT' ones.
        verbose: If False, will only print the name. Else, everything.
    """
    response = client.models.list() # fetches all the models
    for elem in response.data:
        if not all:  
            if "gpt" in elem.id:
                print(elem.id)
        if all:
            print(elem.id)

def print_gptconv_nicely(gpt_conversation: str, system_message: bool = True) -> None:
    """
    Prints a string format GPT conversation (after being extracted from DB) in a human-friendly way.
    Assumes there is a system message. 
    """
    print(get_gptconv_readable_format(gpt_conversation, system_message))

# For local tests
def print_len_token_price(file_path_or_text, Embed = False):
    """
    Basic function to print out the length, the number of token, of a given file or text.
    Chat gpt-3.5-turbo is at $0.002 per 1K token while Embedding is at $0.0004 per 1K tokens. If not specified, we assume it's Chat gpt-3.5-turbo.
    """
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

def repair_gpt_conversation(conversation_as_string: str) -> Optional[str]:
    """
    Repair a GPT conversation to ensure we can then json.loads() it.
    Returns the fixed conversation. 
    """
    result = None
    try:
        new_list = []
        user_ = r'{"role": "user", "content":'
        assistant_ = r'{"role": "assistant", "content":'
        len_a = len(assistant_)
        len_u = len(user_)
        while True:
            bal_user = conversation_as_string.find(user_)
            bal_assistant = conversation_as_string.find(assistant_)
            # Cases: over-over / user-over / assistant-over THEN:: user-user / assistant-assistant / user-assistant / assistant-user
            
            # over-over
            if bal_user == -1 and bal_assistant == -1: 
                break
            
            next_bala = conversation_as_string.find(assistant_, bal_assistant + len_a)
            next_balu = conversation_as_string.find(user_, bal_user + len_u)
            
            # assistant-over
            if bal_user == -1 and next_bala == -1:
                content = make_string_json_safe(conversation_as_string[bal_assistant + len_a: conversation_as_string.rfind("}")])
                new_list.append({"role": "assistant", "content":content})
                break
            # user-over
            if bal_assistant == -1 and next_balu == -1:
                content = make_string_json_safe(conversation_as_string[bal_user + len_u: conversation_as_string.rfind("}")])
                new_list.append({"role": "user", "content":content})
                break
            
            # user-user
            if (0 < bal_user < next_balu < bal_assistant or 
                bal_assistant == -1 and 0 < bal_user < next_balu):
                role = "user"
                content = make_string_json_safe(conversation_as_string[bal_user + len_u: conversation_as_string.rfind("}", None, next_balu)])
                conversation_as_string = conversation_as_string[conversation_as_string.rfind("}", None, next_balu):]
            # assistant-assistant
            elif (0 < bal_assistant < next_bala < bal_user or 
                bal_user == -1 and 0 < bal_assistant < next_bala):
                role = "assistant"
                content = make_string_json_safe(conversation_as_string[bal_assistant + len_a: conversation_as_string.rfind("}", None, next_bala)])
                conversation_as_string = conversation_as_string[conversation_as_string.rfind("}", None, next_bala):]
            # user-assistant
            elif 0 < bal_user < bal_assistant:
                role = "user"
                content = make_string_json_safe(conversation_as_string[bal_user + len_u: conversation_as_string.rfind("}", None, bal_assistant)])
                conversation_as_string = conversation_as_string[conversation_as_string.rfind("}", None, bal_assistant):]
            # assistant-user
            elif 0 < bal_assistant < bal_user:
                role = "assistant"
                content = make_string_json_safe(conversation_as_string[bal_assistant + len_a: conversation_as_string.rfind("}", None, bal_user)])
                conversation_as_string = conversation_as_string[conversation_as_string.rfind("}", None, bal_user):]
            else:
                print("issue - weird use case")
                print(bal_assistant, bal_user, next_bala, next_balu)
                print(conversation_as_string)
                return
            new_list.append({"role": role, "content":content})
        try:
            result = json.dumps(new_list)
            json.loads(result)
        except Exception as e:
            log_issue(e, repair_gpt_conversation)
            return
    except Exception as e:
        log_issue(e, repair_gpt_conversation)
    return result

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

def sanitize_bad_gpt_output(gpt_output: str, case = None) -> str:
    """
    Sanitize bad outputs made by GPT according to bad output we already saw.

    Args:
    - case: Optional. Allows to add other checks that are specifics to the use case.

    Returns:
    - str: The cleaned gpt_output - always strip() the input

    Note:
    - Currently has: 'spar_client', 'spar_sales' and 'spar' as optional filters
    """
    # Check for starting with the assistant prefixes
    if gpt_output.startswith(("Assistant: ", "assistant: ")):
        gpt_output = gpt_output[11:]
    if case in ['spar_client', 'spar']:
        if gpt_output.startswith(("Client: ", "client: ")):
            gpt_output = gpt_output[8:]
    if case in ['spar_sales', 'spar']:
        if gpt_output.startswith(("Salesperson: ", "salesperson: ")):
            gpt_output = gpt_output[13:]
    # Check for starting and ending with single or double quotes
    if (gpt_output.startswith("'") and gpt_output.endswith("'")) or (gpt_output.startswith('"') and gpt_output.endswith('"')):
        gpt_output = gpt_output[1:-1]
    return gpt_output.strip()

def self_affirmation_role(role_chatbot_in_text: str) -> str:
    """
    Function to transform an instruction of the system prompt into a self-affirmation message.

    Theory is that seeing the message twice will make the LLM believe it more.
    """
    clean_text = role_chatbot_in_text.strip()
    clean_text = clean_text.replace(" you are ", " I am ").replace(" You are ", " I am ").replace(" You Are ", " I Am ")
    clean_text = clean_text.replace("You ", "I ").replace(" you ", " I ").replace(" You ", " I ")
    clean_text = clean_text.replace("Your ", "My ").replace(" your ", " my ").replace(" Your ", " My ")
    return clean_text

# *************************************************************************************************
# ****************************************** REGULAR API CALLS ************************************
# *************************************************************************************************

def ask_question_gpt(question:str, role:str = "", model:str = MODEL_CHAT, max_tokens:int = MAX_TOKEN_OUTPUT_DEFAULT, verbose:bool = True, temperature=0, top_p=1, json_on: bool = False) -> str:
    """
    Queries an OpenAI GPT model (GPT-3.5 Turbo / GPT-4 / GPT-4O) with a specific question.

    Args:
        question (str): The question to ask the model.
        role (str, optional): System prompt to be initialized in the chat table, defining ChatGPT's behavior.
        model (str, optional): The model to use. Defaults to GPT-3.5 Turbo. To choose GPT-4O, use 'MODEL_GPT4O' or call 'ask_question_gpto'.
        max_tokens (int, optional): Maximum number of tokens for the answer.
        verbose (bool, optional): Will print information in the console.
        json_on (bool, optional): Whether to force the output in JSON format // UNUSED FOR NOW

    Returns:
        str: The model's reply to the question.
    """
    max_token_window = {
        MODEL_GPT4_TURBO: MAX_TOKEN_WINDOW_GPT4_TURBO - WINDOW_BUFFER,
        MODEL_GPT4O: MAX_TOKEN_WINDOW_GPT4_TURBO - WINDOW_BUFFER,
        MODEL_GPT4_STABLE: MAX_TOKEN_WINDOW_GPT4 - WINDOW_BUFFER,
        MODEL_CHAT: MAX_TOKEN_WINDOW_GPT35_TURBO - WINDOW_BUFFER,
        MODEL_CHAT_STABLE: MAX_TOKEN_WINDOW_GPT35_TURBO - WINDOW_BUFFER,
    }.get(model, MAX_TOKEN_WINDOW_OLD - WINDOW_BUFFER)
    initial_token_usage = calculate_token(role) + calculate_token(question)
    if initial_token_usage > max_token_window:
        print("Your input is too large for the query regardless of the max_tokens for the reply.")
        return ""
    elif initial_token_usage + max_tokens > max_token_window:
        max_tokens_adjusted = max_token_window - initial_token_usage
        print(f"Your input + the requested tokens for the answer exceed the maximum amount of {max_token_window}.\n Please adjust the max_tokens to a MAXIMUM of {max_tokens_adjusted}")
        return ""
    current_chat = initialize_role_in_chatTable(role)
    current_chat = add_content_to_chatTable(question, "user", current_chat)
    if verbose:
        print(f"Completion ~ {max_tokens} tokens. Request ~ {initial_token_usage} tokens.\nContext provided to GPT is:\n{current_chat}")
    return request_chatgpt(current_chat, max_tokens=max_tokens, model=model, temperature=temperature,top_p=top_p, json_on=json_on)

def ask_question_gpt4(question: str, role: str, model=MODEL_GPT4_TURBO, max_tokens=MAX_TOKEN_OUTPUT_DEFAULT_HUGE, verbose = False, temperature=0, top_p=1, json_on=False) -> str:
    """
    Queries Chat GPT 4 with a specific question if too lazy to change the param in ask_question_gpt)
    """
    return ask_question_gpt(question = question, role = role, model = model, max_tokens= max_tokens, verbose=verbose, temperature=temperature, top_p=top_p, json_on=json_on)

def ask_question_gpto(question: str, role: str, model=MODEL_GPT4O, max_tokens=MAX_TOKEN_OUTPUT_DEFAULT_HUGE, verbose = False, temperature=0, top_p=1, json_on=False) -> str:
    """
    Queries Chat GPTO with a specific question if too lazy to change the param in ask_question_gpt)
    """
    return ask_question_gpt(question = question, role = role, model = model, max_tokens= max_tokens, verbose=verbose, temperature=temperature, top_p=top_p, json_on=json_on)

def embed_text(text:str, max_attempts:int=3, model=MODEL_EMB_LARGE) -> Optional[list[float]]:
    """
    Micro function which returns the embedding of one chunk of text or 0 if issue.
    Used for the multi-threading.

    Model is the new large new embedding one. Use Small if speed is a concern.
    Returns None if issue.
    """
    try:
        if text == "": return
        if not isinstance(text, str):
            log_warning("You need to input a string", embed_text, f"You inputed {type(text)}")
            return
        attempts = 0
        while attempts < max_attempts:
            try:
                res = client.embeddings.create(
                    model=model,
                    input=text,
                    encoding_format="float"
                    ).data[0].embedding
                return res
            except Exception as e:
                if not check_co():
                    log_warning("Warning: You don't have internet. Embedding will not work")
                    return
                attempts += 1
                log_warning(f"We faced {e} * Attempt: #{attempts}/ 3", embed_text)
        log_issue(f"No answer despite {max_attempts} attempts", embed_text, f"This was the text: {text[:100]}")
    except Exception as e:
        log_issue(e, embed_text, f"""For text {text[:300] + ('...' if len(text)> 300 else '')}""")

def request_chatgpt(current_chat: list, max_tokens: int, stop_list=False, max_attempts=3, model=MODEL_CHAT, temperature=0, top_p=1, json_on=False) -> str:
    """
    Calls the ChatGPT OpenAI completion endpoint with specified parameters.

    Args:
        current_chat (list): The prompt used for the request.
        max_tokens (int, optional): Maximum number of tokens for the answer.
        stop_list (bool, optional): Whether to use specific stop tokens. Defaults to False.
        max_attempts (int, optional): Maximum number of retries. Defaults to 3.
        model (str, optional): ChatGPT OpenAI model used for the request. Defaults to the GPT-3.5 Turbo
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
    rep = OPEN_AI_ISSUE
    #print("Writing the reply for ", current_chat) # Remove in production - to see what is actually fed as a prompt
    while attempts < max_attempts and not valid:
        try:
            response = client.chat.completions.create(
                messages= current_chat,
                temperature=temperature,
                max_tokens= int(max_tokens),
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
                model= model,
            )
            rep = response.choices[0].message.content
            rep = rep.strip()
            valid = True
        except Exception as e:
            attempts += 1
            error_message = str(e)
            if 'Rate limit reached' in error_message:
                print(f"Rate limit reached. We will slow down and sleep for 300ms. This was attempt number {attempts}/{max_attempts}")
                time.sleep(0.3)
            else:
                print(f"Error. This is attempt number {attempts}/{max_attempts}. The exception is {e}. Trying again")
            if {attempts} == 2:
                print(f"Trying with the previous model: {MODEL_CHAT_BACKUP}")
                model = MODEL_CHAT_BACKUP
    if rep == OPEN_AI_ISSUE and check_co():
        print(f" ** We have an issue with Open AI using the model {model}")
        log_issue(f"No answer despite {max_attempts} attempts", request_chatgpt, "Open AI is down")
    return rep
    
# *************************************************************************************************
# *************************************************************************************************

if __name__ == "__main__":
    pass