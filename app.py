from utils import (
    from_storages,
    # get_vector_tool
    # get_comparision_tool
    # get_negotiation_tool
    router_query_engine
)
import os
import gradio as gr

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
)
from llama_index.core.agent import ReActAgent

# For exponential backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


# Make sure to: export OPENAI_API_KEY=<Your OpenAI key>
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GOOGLE_API_KEY')

Settings.llm = Gemini(api_key=gemini_api_key, model="models/gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding(
    api_key=gemini_api_key, model="models/embedding-001")

vector_index, summary_index = from_storages()

# vector_tool = get_vector_tool(vector_index=vector_index)
# comparison_tool = get_comparision_tool(vector_index=vector_index)
# negotiate_tool = get_negotiation_tool(vector_index=vector_index)

# llm = OpenAI(model="gpt-4o-mini")
# agent = ReActAgent.from_tools(
#     tools=[
#         vector_tool,
#         comparison_tool,
#         negotiate_tool,
#     ],
#     verbose=True,
#     llm=llm,
#     context='This agent assists users with finding apartments, comparing options, and negotiating offers.'
# )

query_engine = router_query_engine(vector_index, summary_index)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_with_sources(query):
    """
    Query the index and display the answer along with source information.

    Args:
    query_engine: The LlamaIndex query engine to use
    query (str): The query string

    Returns:
    None: Prints the results directly
    """
    # Query the index
    # answer = agent.chat(query)
    answer = query_engine.query(query)

    divider = "-" * 50

    references = ''
    if (len(answer.source_nodes) > 0):
        for node in answer.source_nodes:
            # Extract metadata from the node
            name = node.node.metadata.get('Name', 'Name not found')
            price = node.node.metadata.get('Price', 'Price not found')
            location = node.node.metadata.get('Location', 'Location not found')
            url = node.node.metadata.get('URL', 'URL not found')
            description = node.node.metadata.get(
                'Description', 'Description not found')
            area = node.node.metadata.get('Area', 'Area not found')

            references = references + f"<li>{name}</li>"
            references = references + f"<li>Giá: {price}</li>"
            references = references + f"<li>Khu vực: {location}</li>"
            references = references + f"<li>URL: {url}</li>"
            references = references + f"<li>Mô tả: {description}</li>"
            references = references + f"<li>Diện tích: {area}</li>"
            references = references + divider + "<br>"

        response = f"<pre style='text-wrap: pretty;'><p>{
            answer}</p><br><b>Tham khảo thêm:</b><br><ul>{references}</ul></pre>"
    else:
        response = f"<pre style='text-wrap: pretty;'><p>{
            answer}</p></pre>"

    return response


# Create Gradio Interface
iface = gr.Interface(
    fn=query_with_sources,
    inputs="text",
    outputs=gr.HTML(),
    title="Chatbot tư vấn cho thuê căn hộ chung cư",
    description="Chatbot tư vấn cho thuê căn hộ chung cư ở các quận/huyện Tp HCM"
)

# Launch the interface
iface.launch()
