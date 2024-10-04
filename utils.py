from web_crawler import get_doc_nodes

from typing import List, Dict, Optional

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

from llama_index.core import (
    Settings,
    PromptTemplate,
    SummaryIndex,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore

import chromadb
import mlflow

db_path = "./bds_db"
# Set up ChromaDB client
db = chromadb.PersistentClient(path=db_path)


def init_index(name):
    docstore_path = f"./{name}_docstore.json"
    collection_name = f"{name}_collection"

    docstore = SimpleDocumentStore()
    nodes = []

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
        nodes = get_doc_nodes(name)
        # add note to docstore for BM25
        docstore.add_documents(nodes)

    # Create or get the ChromaDB collection
    chroma_collection = db.get_or_create_collection(collection_name)
    # Initialize a ChromaVectorStore and store the nodes
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

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


def get_vector_engine(vector_index):
    vector_engine = vector_index.as_retriever(similarity_top_k=2)
    bm25_retriever = QueryFusionRetriever(
        [
            vector_engine,
            BM25Retriever.from_defaults(
                docstore=vector_index.docstore, similarity_top_k=2
            ),
        ],
        # query_gen_prompt
        # num_queries=1,
    )
    query_engine = RetrieverQueryEngine(bm25_retriever)

    return query_engine


def get_summary_tool(summary_index, tool_name):
    summary_engine = summary_index.as_retriever(
        response_mode="tree_summarize",
        use_async=True,
    )
    bm25_summary_retriever = QueryFusionRetriever(
        [
            summary_engine,
            BM25Retriever.from_defaults(
                docstore=summary_index.docstore, similarity_top_k=2
            ),
        ],
        # query_gen_prompt
        # num_queries=1,
    )
    tool_name = f"summary_tool_{tool_name.replace('-', '_')}"
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{tool_name}",
        query_engine=RetrieverQueryEngine(bm25_summary_retriever),
        description=(
            "Useful for summarize context"
        ),
    )

    return summary_tool


def router_query_engine(vector_index, summary_index):
    summary_tool = get_summary_tool(summary_index=summary_index)

    vector_engine = get_vector_engine(vector_index=vector_index)
    vector_tool = QueryEngineTool.from_defaults(
        name=f"vector_tool",
        query_engine=vector_engine,
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


def get_vector_tool(vector_index, tool_name):
    def vector_query(
        query: str,
        # propterty detail
        name: Optional[str] = None,
        price: Optional[str] = None,
        location: Optional[str] = None,
        square_footage: Optional[str] = None,
        property_type: Optional[str] = None,
        bedrooms: Optional[str] = None,
        bathrooms: Optional[str] = None,
        amenities: Optional[str] = None,
        year_build: Optional[str] = None,
        floor_level: Optional[str] = None,
        # user preferences
        user_budget: Optional[str] = None,
        preferred_location: Optional[str] = None,
        transportation:  Optional[str] = None,
    ) -> str:
        name = price or ""
        price = price or "0"
        location = location or ""
        square_footage = square_footage or "0"
        property_type = property_type or ""
        bedrooms = bedrooms or "0"
        bathrooms = bathrooms or "0"
        amenities = amenities or ""
        year_build = year_build or "0"
        floor_level = floor_level or "",
        # user preferences
        user_budget = user_budget or "0"
        preferred_location = preferred_location or ""
        transportation = transportation or ""

        metadata_dicts = [
            {"key": "Name", "value": name},
            {"key": "Price", "value": price},
            {"key": "Location", "value": location},
            {"key": "SquareFootage", "value": square_footage},
            {"key": "PropertyType", "value": property_type},
            {"key": "Bedrooms", "value": bedrooms},
            {"key": "Bathrooms", "value": bathrooms},
            {"key": "Amenities", "value": amenities},
            {"key": "YearBuild", "value": year_build},
            {"key": "FloorLevel", "value": floor_level},
            # user preferences
            {"key": "UserBudget", "value": user_budget},
            {"key": "PreferredLocation", "value": preferred_location},
            {"key": "Transportation", "value": transportation},
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
            # query_gen_prompt=PromptTemplate("")
            # num_queries=1,
        )

        query_engine = RetrieverQueryEngine(bm25_retriever)
        response = query_engine.query(query)

        # prompt_template = """\
        # Please write a passage to answer the question: {query_str}

        # Try to include as many key details as possible.

        # Passage: """
        # prompt = PromptTemplate(
        #     input_variables=["question"], template=prompt_template)

        # hyde_query_engine = TransformQueryEngine(
        #     query_engine,
        #     query_transform=HyDEQueryTransform(
        #         hyde_prompt=prompt,
        #         include_original=True,
        #     )
        # )

        # response = hyde_query_engine.query(query)
        return response

    # Information Retrieval Agent
    tool_name = f"vector_tool_{tool_name.replace('-', '_')}"
    vector_tool = FunctionTool.from_defaults(
        name=tool_name,
        fn=vector_query,
        # description=("")
    )
    return vector_tool


def get_comparision_tool(vector_index):
    # load LLM from global setting
    llm = Settings.llm

    query_engine = get_vector_engine(vector_index=vector_index)

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
        description=(
            "Compares selected apartments based on user-defined criteria."),
        fn=compare_query,
    )
    return comparison_tool


def get_negotiation_tool(vector_index):
    # load LLM from global setting
    llm = Settings.llm

    query_engine = get_vector_engine(vector_index=vector_index)
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
        description=(
            "Helps negotiate with landlords by suggesting strategies and counteroffers"),
        fn=negotiate_query,
    )
    return negotiate_tool
