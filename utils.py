import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from markdownify import markdownify as md

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine import TransformQueryEngine, RetrieverQueryEngine
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from web_reader import SimpleWebPageReader
# from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import (
    Settings,
    PromptTemplate,
    Document,
    SummaryIndex,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore

import chromadb

# Constants
BASE_URL = "https://nhadat.cafeland.vn/cho-thue/can-ho-chung-cu-tai-tp-ho-chi-minh"
DEFAULT_TIMEOUT = 50
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Default headers for requests
HEADERS = {
    "User-Agent": USER_AGENT
}

db_path = "./bds_db"
docstore_path = "./docstore.json"
collection_name = "bds_collection"
# Set up ChromaDB client
db = chromadb.PersistentClient(path=db_path)

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

        if current_page > 100 or not next_button:
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


def from_storages():
    docstore = SimpleDocumentStore()
    nodes = []

    # Create or get the ChromaDB collection
    chroma_collection = db.get_or_create_collection(collection_name)
    # Initialize a ChromaVectorStore and store the nodes
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Check if the collection already exists
    collections = db.list_collections()
    collection_names = [col.name for col in collections]

    if collection_name in collection_names:
        print(f"Collection '{
            collection_name}' already exists. Skipping document processing.")

        # load docstore if it's exists
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
    else:
        print(f"Collection '{
            collection_name}' does not exist. Creating new collection and processing documents.")

        # Run your function to get nodes from documents only if the collection doesn't exist
        nodes = get_doc_nodes()
        # add note to docstore for BM25
        docstore.add_documents(nodes)

    # Create a StorageContext
    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store)
    # save docstore
    if (len(nodes) > 0):
        storage_context.docstore.persist(docstore_path)

    # Create a VectorStoreIndex
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    summary_index = SummaryIndex(nodes, storage_context=storage_context)

    return vector_index, summary_index

# Information Retrieval Agent


def get_summary_tool(summary_index):
    summary_engine = summary_index.as_retriever(
        response_mode="tree_summarize",
        use_async=True,
    )
    bm25_retriever = QueryFusionRetriever(
        [
            summary_engine,
            BM25Retriever.from_defaults(
                docstore=summary_index.docstore, similarity_top_k=2
            ),
        ],
        num_queries=1,
        use_async=True,
    )

    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool",
        query_engine=RetrieverQueryEngine(bm25_retriever),
        description=(
            "Useful for summarize context"
        ),
    )
    return summary_tool


def get_vector_tool(vector_index):
    def vector_query(
        query: str,
        price: Optional[str] = None,
        location: Optional[str] = None,
        area: Optional[str] = None,
    ) -> str:
        price = price or "0"
        location = location or ""
        area = area or "0"
        metadata_dicts = [
            {"key": "Price", "value": price},
            {"key": "Location", "value": location},
            {"key": "Area", "value": area},
        ]
        vector_engine = vector_index.as_retriever(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )

        bm25_retriever = QueryFusionRetriever(
            [
                vector_engine,
                BM25Retriever.from_defaults(
                    docstore=vector_index.docstore, similarity_top_k=2
                ),
            ],
            num_queries=1,
            use_async=True,
        )
        query_engine = RetrieverQueryEngine(bm25_retriever)

        hyde = HyDEQueryTransform(
            include_original=True
        )
        with_hyde = TransformQueryEngine(
            query_engine, query_transform=hyde)

        response = with_hyde.query(query)
        return response

    # Information Retrieval Agent
    vector_tool = FunctionTool.from_defaults(
        name=f"vector_tool",
        fn=vector_query,
    )
    return vector_tool


def get_comparision_tool(vector_index):
    # load LLM from global setting
    llm = Settings.llm

    vector_engine = vector_index.as_retriever(similarity_top_k=2)
    bm25_retriever = QueryFusionRetriever(
        [
            vector_engine,
            BM25Retriever.from_defaults(
                docstore=vector_index.docstore, similarity_top_k=2
            ),
        ],
        num_queries=1,
        use_async=True,
    )
    query_engine = RetrieverQueryEngine(bm25_retriever)

    # Comparison Agent
    def compare_query(
        criteria,
        apartment_ids: Optional[List[str]] = None,
    ):
        apartment_ids = apartment_ids or []
        # Query the index to retrieve information about the selected apartments
        apartments_info = []
        for apartment_id in apartment_ids:
            # Here we assume you have a way to query or filter documents by apartment_id
            query_result = query_engine.query(
                f"Details of apartment with ID {apartment_id}")
            apartments_info.append(query_result)

        # Combine the information into a formatted string
        apartments_info_str = "\n\n".join(
            [f"Apartment {i+1}: {info}" for i, info in enumerate(apartments_info)])

        comparison_prompt = PromptTemplate(
            "Compare the following apartments based on these criteria: {criteria}.\n\n"
            "{apartments_info_str}\n\nProvide a detailed comparison."
        )

        comparison = llm.predict(
            comparison_prompt, criteria=criteria, apartments_info_str=apartments_info_str)

        return comparison

    comparison_tool = FunctionTool.from_defaults(
        name=f"comparison",
        fn=compare_query,
        # description="Compares selected apartments based on user-defined criteria."
    )
    return comparison_tool


def get_negotiation_tool(vector_index):
    # load LLM from global setting
    llm = Settings.llm

    vector_engine = vector_index.as_retriever(similarity_top_k=2)
    bm25_retriever = QueryFusionRetriever(
        [
            vector_engine,
            BM25Retriever.from_defaults(
                docstore=vector_index.docstore, similarity_top_k=2
            ),
        ],
        num_queries=1,
        use_async=True,
    )
    query_engine = RetrieverQueryEngine(bm25_retriever)
    # Negotiation Agent

    def negotiate_query(
        landlord_offer,
        target_price
    ):
        # Retrieve market data to support the counteroffer
        market_data = query_engine.query(
            "Market data for similar apartments in the area")

        counteroffer_prompt = PromptTemplate(
            "Based on the market data: {market_data}, "
            "suggest a reasonable counteroffer for the target price of {target_price}."
        )
        # Generate the negotiation strategy and counteroffer using the LLM
        counteroffer_response = llm.predict(
            counteroffer_prompt, market_data=market_data, target_price=target_price)

        strategy_prompt = PromptTemplate(
            "The landlord is offering a rental price of {landlord_offer}. "
            "The tenant is aiming for a rental price of {target_price}. "
            "Suggest a negotiation strategy that the tenant can use."
        )

        # Generate the negotiation strategy and counteroffer using the LLM
        negotiation_response = llm.predict(
            strategy_prompt, landlord_offer=landlord_offer, target_price=target_price)

        return negotiation_response, counteroffer_response

    negotiate_tool = FunctionTool.from_defaults(
        name=f"negotiation_strategy",
        fn=negotiate_query,
        # description="Helps negotiate with landlords by suggesting strategies and counteroffers"
    )
    return negotiate_tool


def router_query_engine():
    vector_index, summary_index = from_storages()

    summary_tool = get_summary_tool(summary_index)

    vector_engine = vector_index.as_retriever(
        similarity_top_k=2,
    )

    bm25_retriever = QueryFusionRetriever(
        [
            vector_engine,
            BM25Retriever.from_defaults(
                docstore=vector_index.docstore, similarity_top_k=2
            ),
        ],
        num_queries=1,
        use_async=True,
    )

    vector_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool",
        query_engine=RetrieverQueryEngine(bm25_retriever),
        description=(
            "Useful for summarize context"
        ),
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            vector_tool,
            summary_tool,
        ],
        verbose=True
    )

    return query_engine
