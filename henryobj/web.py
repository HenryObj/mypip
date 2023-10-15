"""
    @Author:				Henry Obegi <HenryObj>
    @Email:					hobegi@gmail.com
    @Creation:				Friday 10th of February
    @LastModif:             Sunday 10th of October 2023
    @Filename:				web.py
    @Purpose                Functions related to fetching content from the web
    @Partof                 pip package
"""

# ************** IMPORTS ****************

from .base import *

from bs4 import BeautifulSoup
import concurrent.futures
from collections import deque
from requests.adapters import HTTPAdapter
from requests.exceptions import SSLError
from urllib3.util.retry import Retry
from urllib.parse import urljoin

# ****** PATHS & GLOBAL VARIABLES *******

HTTP_URL_PATTERN = r'^http[s]?://.+'   # Regex pattern to match a URL
HTTP_STRICT_URL_PATTERN = r'https?://[\\w/:%#\\$&\\?\\(\\)~\\.=\\+\\-]+'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
}

# ****** SET UP SESSION OBJECT FOR SCRAPPING *******

'''
    500: Internal Server Error, which indicates that the server encountered an error while processing the request.
    502: Bad Gateway, which indicates that the server, while acting as a gateway or proxy, received an invalid response from the upstream server it accessed in attempting to fulfill the request.
    504: Gateway Timeout, which indicates that the server, while acting as a gateway or proxy, did not receive a timely response from the upstream server.
    These status codes are chosen because they represent temporary issues that may be resolved on subsequent requests.
'''

def create_session(max_retries: int = 3, backoff_factor: float = 0.3, status_forcelist: tuple = (500, 502, 504)) -> requests.Session:
    """
    Create and configure a requests.Session object.
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,  # Set the backoff factor for exponential delay
        status_forcelist=status_forcelist, # Set the list of HTTP status codes to consider for retries
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

session = create_session()

# ****************** FUNCS ******************

# Removes about 60% of the content
def clean_soup(soup: BeautifulSoup, url: Optional[str] = None) -> str:
    """
    Clean and extract text from a BeautifulSoup object.

    Args:
        soup (BeautifulSoup): The parsed HTML or XML document to be cleaned.
        url (Optional): Default is None. Allows to have special rules for special websites.

    Returns:
        str: The cleaned text content extracted from the soup.
    """
    if ("You need to enable JavaScript to run this app." in soup.get_text()):
        # Here, we would need to use selenium and do a headless browser
        log_issue("Couldn't get the data of a wepage - JS needed", clean_soup)
    
    # Performance tracker
    print(" * Before cleaning:     ", end ="")
    print_len_token_price(soup.get_text())
    
    # Normal cleaner - we keep the meta tag as it creates a big loss for Wikipedia
    for a in soup.find_all('a'):  
        if not is_useful_link(a):
            a.decompose()
    tags_to_decompose = ["header", "script", "nav", "style", "popup", "footer", "button", "form", "link", "img", "video"]
    for tag in soup(tags_to_decompose):
        tag.decompose()
    for tag in soup(lambda tag: tag.has_attr("aria-hidden") and tag["aria-hidden"] == "true"):
        tag.decompose()
    
    # special cases can be listed with elif here
    remove_long = False
    if "wikipedia.org" in url:
        remove_citations(soup)
        remove_long = True
    else:
        remove_reviews(soup)
    
    text = soup.get_text()
    if remove_long: text = remove_long_sentences(text)
    text = remove_non_printable(text)
    text = remove_excess(text)

    # Performance tracker
    print(" ** After cleaning:     ", end ="*")
    print_len_token_price(text)

    return text

def clean_url_to_filename(url: str) -> str:
    """
    Convert URL to a suitable filename.
    """
    file_name = urlparse(url).path
    if file_name == "": file_name = "home_landing_none"
    file_name = file_name.replace("/","_")
    file_name = file_name.replace("-","_")
    if file_name.startswith("_"): file_name = file_name[1:]
    return file_name

def clean_url_into_title(url: str) -> str:
    """
    Convert URL to a shortened version excluding the protocol and 'www'.

    Examples:
    "https://www.skyla.chat" will return "skyla.chat/"
    "https://chat.openai.com/sub1/sub3" will return "chat.openai.com/sub1/sub3"

    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace('www.', '')
    path = parsed_url.path.strip('/')
    return f"{domain}/{path}"

def crawl_website(url: str, how_many_pages = 30, memory_store = None) -> dict:
    """
    Crawl website starting from a given URL. Stores data in-memory. Return the dictionnary with all the content.
    """
    submitted = 0
    if not memory_store:
        memory_store = {}  # In-memory storage for crawled data
    local_domain = urlparse(url).netloc 
    queue = deque([url])
    seen = set([url])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while queue:
            url = queue.pop()
            if not check_valid_url(url):
                continue
            response = session.head(url, headers=HEADERS)
            content_type = response.headers.get("content-type", "")
            if not content_type_is_text(content_type):
                continue
            data_name = clean_url_into_title(url)
            future = executor.submit(fetch_content_url, url)
            submitted += 1
            if submitted == how_many_pages:
                break
            future.add_done_callback(lambda future: wrap_handle_fetch_result(future, data_name, memory_store))
            for link in fetch_domain_links(local_domain, url):
                if link not in seen:
                    queue.append(link)
                    seen.add(link)
    return memory_store

def fetch_hyperlinks(url: str) -> List[str]:
    """
    Fetch and return all hyperlinks from a given URL.
    """
    try:
        response = requests.get(url, headers=HEADERS)
        # If the response is not HTML, return an empty list
        if not response.headers.get('Content-Type', '').startswith('text/html'):
            return []
        html = response.text
    except Exception as e:
        log_issue(e, fetch_hyperlinks, f"Couldn't fetch the links of url {url}")
        return []
    soup = BeautifulSoup(html, 'html.parser')
    links = [link.get('href') for link in soup.find_all('a')]
    return links

# Fetch URL - works as a standalone
# Might want to test the driver version with selenium - driver = webdriver.Firefox()
def fetch_content_url(url: str, attempt: int = 0) -> Optional[str]:
    """
    Fetch and clean content from a webpage.
    """
    print(f"Doing {url}") # @ to be removed when prod
    try:
        data = session.get(url, timeout=5)  # Using session object instead of requests
        if data.status_code == 200:
            soup = BeautifulSoup(data.text, "html.parser")
            clean = clean_soup(soup, url)
            return clean
        elif data.status_code == 429 and attempt < 2:
            # in case of too many requests (429 is rate limiting), we wait and attempt again using exponential backoff
            attempt += 1
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)
            return fetch_content_url(url, attempt)
        else:
            # "URL could not be accessed:" - @ ToDecide if we want to do smth with it
            return None
    except SSLError as e:
        log_issue(e, fetch_content_url, f"SSL/TLS error for url {url}")
        return None
    except Exception as e:
        log_issue(e, fetch_content_url, f"For url {url}")
        return None

def is_useful_link(tag):
    '''
    Mini function to check if a link is surrounded with content, hence useful, or alone.
    Return True if useful, False otherwise.
    '''
   # Check previous and next siblings
    prev_sibling = tag.previous_sibling
    next_sibling = tag.next_sibling
    if prev_sibling and prev_sibling.string and prev_sibling.string.strip() != '':
        return True
    if next_sibling and next_sibling.string and next_sibling.string.strip() != '':
        return True
    # Check if link is in the middle of the parent's text (excluding the link's own text)
    parent = tag.parent
    if parent:
        text_without_link = parent.get_text().replace(tag.get_text(), '').strip()
        if text_without_link:
            return True
    return False

def fetch_domain_links(local_domain, url):
    '''
    Returns the list of all unique urls of a given domain. Search start from a specific page (url).
    Doesn't return links that are not part of the domain.
    '''
    clean_links = set()
    try:
        raw_links = fetch_hyperlinks(url)
        if raw_links is None:
            print("No hyperlink", fetch_domain_links, f"For {url} and {local_domain}")
            return []
        for link in set(raw_links):
            if link is None:
                continue
            valid_link = False
            # If the link is a URL, check if it is within the same domain
            if re.search(HTTP_URL_PATTERN, link):
                if urlparse(link).netloc == local_domain: # to check that the domain is the same
                    valid_link = link
            # If the link is not a URL, check if it is a relative link
            else:
                if link.startswith("/") and not link.startswith("//"):
                    link = link[1:]
                    valid_link = f"https://{local_domain}/{link}"
                elif link.startswith("#") or link.startswith("mailto:"):
                    continue
                else:
                    valid_link = urljoin(url, link)            
            if valid_link:
                if valid_link.endswith("/"):
                    valid_link = valid_link[:-1]
                clean_links.add(valid_link)
    except Exception as e:
        log_issue(e, fetch_domain_links, f"For {url} and {local_domain}")
    return clean_links

def remove_citations(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Remove citation tags from a BeautifulSoup object.
    """
    for tag in soup.find_all(lambda t: t.has_attr('id') and 'cite_note' in t['id']):
        tag.decompose()
    return soup

def remove_long_sentences(text: str, max_words: int = 50) -> str:
    """
    Remove sentences from the text that have more than max_words words.

    Args:
        text (str): The input text.
        max_words (int, optional): The maximum number of words a sentence can have. Defaults to 50.

    Returns:
        str: Text with long sentences removed.
    """
    # Use regex to split text into sentences
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    cleaned_sentences = [sentence for sentence in sentences if len(sentence.split()) <= max_words]
    return ' '.join(cleaned_sentences)

def remove_reviews(soup : BeautifulSoup) -> BeautifulSoup:
    '''
    Slightly risky function which removes the reviews from a Shopify store.
    '''
    pattern = re.compile(r'data-verified-.*|review.*')
    for div in soup.find_all('div'):
        if div.attrs is not None: 
            div_class = div.get('class')
            if div_class is not None and any(pattern.match(class_) for class_ in div_class):
                div.decompose()
    return soup

def wrap_handle_fetch_result(future, data_name, memory_store):
    '''
    Wrapper to allow adding paramaters to the callback function.
    '''
    crawl_handle_fetch_result(future, data_name, memory_store)

def crawl_handle_fetch_result(future, data_name, memory_store) -> None:
    """
    Handles fetched data and stores it in-memory.
    """
    content = future.result()
    if content is not None:
        memory_store[data_name] = content

def content_type_is_text(content_type):
    '''
    Used to skip if content_type.startswith("image") or content_type.startswith("application/pdf").
    Now, we skip if it's not text or xml.
    '''
    if content_type.startswith("text") or content_type.startswith("application/xhtml+xml"):
        return True
    else:
        return False


# *************************************************************
if __name__ == "__main__":
    pass