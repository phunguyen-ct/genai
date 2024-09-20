import requests
from bs4 import BeautifulSoup

from typing import List, Dict, Optional
from markdownify import markdownify as md

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import (
    Document,
)

from web_reader import SimpleWebPageReader


# Constants
BASE_URL = "https://nhadat.cafeland.vn/cho-thue/can-ho-chung-cu-tai-tp-ho-chi-minh"
DEFAULT_TIMEOUT = 50
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Default headers for requests
HEADERS = {
    "User-Agent": USER_AGENT
}

LIMIT = 10

# Crawl the data from website


def crawl_webpage(url: str, headers: dict = HEADERS, timeout: int = DEFAULT_TIMEOUT) -> Optional[List[Dict[str, str]]]:
    """
    Fetch the webpage, parse it, and extract data.

    Args:
        url (str): The URL of the webpage to fetch.
        headers (dict, optional): HTTP headers to send with the request. Defaults to HEADERS.
        timeout (int, optional): Timeout for the request in seconds. Defaults to DEFAULT_TIMEOUT.

    Returns:
        Optional[List[Dict[str, str]]]: A list of dictionaries containing data, or None if an error occurred.
    """
    bds_card_data = []
    current_page = 1
    while True:
        # Send GET request to the URL
        request_url = f'{url}/page-{current_page}'
        print('request_url', request_url)
        response = requests.get(request_url, headers=headers, timeout=timeout)
        # if response.status_code != 200:
        #     break

        print(f"Status Code: {response.status_code}")

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all cards
        bds_cards = soup.find_all('div', class_='row-item')

        # Extract data from each card
        for card in bds_cards:
            bds_info_element = card.find('div', class_='info-real')
            if (bds_info_element == None):
                continue

            # Extract name
            name = bds_info_element.find(
                'div', class_='reales-title').find('a')['title']
            bds_url = bds_info_element.find(
                'div', class_='reales-title').find('a')['href']

            # Extract price
            price_element = bds_info_element.find(
                'div', class_='reales-info-general')
            price = price_element.find(
                'span', class_='reales-price').text.strip()
            area = price_element.find(
                'span', class_='reales-area').text.strip()

            # Extract location & description
            location_element = bds_info_element.find(
                'div', class_='reales-info-general').find('div', class_='info-location')
            location = location_element.get_text(separator=' ', strip=True)

            description = bds_info_element.find(
                'div', class_='reales-preview').text.strip()

            # Append extracted data to the list
            bds_card_data.append({
                'Name': name,
                'Price': price,
                'URL': bds_url,
                'Location': location,
                'Description': description,
                'Area': area,
            })

        next_button = None
        paging = soup.find('ul', class_='pagination').find_all('li')
        for page_button in paging:
            page_str = page_button.find('a').text.strip()
            if (page_str == '»'):
                next_button = page_str

        if current_page > LIMIT or not next_button:
            break

        # Increment page counter
        current_page += 1

    return bds_card_data


# Process the raw data (from html to markdown)
def extract_documents(bds_data: List[Dict[str, str]]) -> List[Document]:
    """
    Extract documents from bds data using SimpleWebPageReader.

    This function takes a list of data and creates Document objects
    for each. It uses SimpleWebPageReader to load the content from
    the provided URLs.

    Args:
        bds_data (List[Dict[str, str]]): List of dictionaries containing data.
            Each dictionary should have 'URL', 'Title', and 'Date' keys.

    Returns:
        List[Document]: List of Document objects created from thes.

    Raises:
        Exception: If there's an error extracting content from a URL.
    """
    reader = SimpleWebPageReader()
    documents = []

    for bds in bds_data:
        url = bds['URL']
        print('content url', url)
        try:
            loaded_documents = reader.load_data([url])
            if loaded_documents:
                doc = loaded_documents[0]
                # Assuming preprocess_text is defined elsewhere
                doc.text = preprocess_text(doc.text)
                doc.metadata.update({
                    'Name': bds['Name'],
                    'Price': bds['Price'],
                    'URL': bds['URL'],
                    'Location': bds['Location'],
                    'Description': bds['Description'],
                    'Area': bds['Area'],
                })
                documents.append(doc)
            else:
                print(f"No content extracted from {url}")
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")

    return documents


def preprocess_text(text: str) -> str:
    """
    Preprocess the extracted text: convert HTML to Markdown, remove unwanted sections,
    clean up the text, and add the full domain to relative image URLs.

    Args:
        text (str): The input HTML text to preprocess.

    Returns:
        str: The preprocessed Markdown text.
    """
    # Convert HTML to Markdown
    markdown_text = md(text, heading_style="ATX")

    # Split the text into lines
    lines = markdown_text.split('\n')

    # Find the index of the first line starting with '#'
    start_index = next((i for i, line in enumerate(lines)
                       if line.startswith('# ')), 0)

    # Find the index of the line "Vị trí tài sản" or the end of the text
    end_index = next((i for i, line in enumerate(
        lines) if line.strip() == "Vị trí tài sản"), len(lines))

    # Process lines between start_index and end_index
    processed_lines = []
    for line in lines[start_index:end_index]:
        # Remove extra whitespace
        line = line.strip()

        if line:
            # print('line', line)
            processed_lines.append(line)

    # Join the processed lines
    processed_text = '\n'.join(processed_lines)

    return processed_text

# A dummy function for display result of markdown to easily demonstate


def display_markdown(text: str, max_length: int = 2000) -> None:
    """
    Display a preview of the Markdown text in Jupyter notebook.

    This function takes a Markdown string and displays it in the notebook,
    truncating it if it exceeds the specified maximum length.

    Args:
        text (str): The Markdown text to display.
        max_length (int, optional): Maximum length of the preview. Defaults to 2000.
    """
    preview = text[:max_length]
    if len(text) > max_length:
        preview += "..."


def get_doc_nodes():
    # Crawl and convert into Dataframe
    bds_card_data = crawl_webpage(BASE_URL, headers=HEADERS)
    print(f"Number of extracted: {len(bds_card_data)}")
    print(f"-"*50)

    # Example usage
    documents = extract_documents(bds_card_data)
    print(f"Created {len(documents)} Document objects.")
    print("-" * 50)
    print("Metadata of the first document:")
    print("-" * 50)
    print(f"Metadata: {documents[0].metadata}")
    print("-" * 50)
    print(documents[0].text[:500] + "...")

    # Parse documents into nodes
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    # Display information about the nodes
    print(f"Number of nodes created: {len(nodes)}")

    # Display metadata and content of a specific node (e.g., the 8th node)
    print("\nMetadata of the 8th node:")
    print(nodes[7].metadata)

    print("\nContent preview of the 8th node:")
    display_markdown(nodes[7].text)

    return nodes
