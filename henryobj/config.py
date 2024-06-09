# PIP Package of utlity functions used in various projects

# Made by Henry Obegi - hobegi@gmail.com
# First version: September 2023
# Last update: 31st of May 2024



# ****** TEXT
OPEN_AI_ISSUE = r"%$144$%" # When OpenAI is down
ERROR_MESSAGE = "An error occurred and was logged"
WARNING_UNKNOWN = "\033[31mUNKNOWN\033[0m"


# ****** For Web
HTTP_URL_PATTERN = r'^http[s]?://.+'   # Regex pattern to match a URL
HTTP_STRICT_URL_PATTERN = r'https?://[\\w/:%#\\$&\\?\\(\\)~\\.=\\+\\-]+'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
}

# ****** TOKEN LIMITATIONS
MAX_TOKEN_OUTPUT = 4096
MAX_TOKEN_OUTPUT_DEFAULT = 300
MAX_TOKEN_OUTPUT_DEFAULT_HUGE = 3000

MAX_TOKEN_WINDOW_OLD = 4096 
MAX_TOKEN_WINDOW_GPT4_TURBO = 128000
MAX_TOKEN_WINDOW_GPT35_TURBO = 16385
MAX_TOKEN_WINDOW_GPT4 = 8192
WINDOW_BUFFER = 150

# ****** MODELS
MODEL_GPT4O = r"gpt-4o"

MODEL_GPT4_TURBO = r"gpt-4-1106-preview" #Max 128,000 token context window total with 4,096 output
MODEL_GPT4_STABLE = r"gpt-4" # 8K context window and 4,096 output

MODEL_CHAT_BACKUP = r"gpt-3.5-turbo-1106" # Context 16,385 tokens - Reply 4,096
MODEL_CHAT = r"gpt-3.5-turbo-0125"
MODEL_CHAT_STABLE = r"gpt-3.5-turbo"

MODEL_OLD = r"text-embedding-ada-002"
MODEL_EMB_LARGE = r"text-embedding-3-large"
MODEL_EMB_SMALL = r"text-embedding-3-small"


# ******* GPT
BUFFER_README_INPUT = 30000
LARGE_INPUT_THRESHOLD = 10000  # Threshold for considering an input as large

# ******* Papertrail
CODE_ERR_PT = r"HO144"
CODE_WARN_PT = r"HO69"
CODE_DAILY_PT = r"HO1989"