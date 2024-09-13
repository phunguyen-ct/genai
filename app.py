from utils import (
    from_storages,
    get_summary_tool,
    get_vector_tool,
    get_comparision_tool,
    get_negotiation_tool,
    router_query_engine
)
import os
import gradio as gr

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import (
    Settings,
    PromptTemplate,
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

# Make sure to: export HF_TOKEN=<Your HuggingFaceAPI key>
hf_token = os.getenv("HF_TOKEN")

hf_llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    query_wrapper_prompt=PromptTemplate(
        "<s>[INST]"
        " Always answer the query using the provided context information, "
        " and not prior knowledge to answer the question: {query_str}\n"
        " Response in Vietnamese.\n"
        "[/INST] </s>\n"
    ),
    token=hf_token,
    generate_kwargs={"temperature": 0.2, "top_k": 5, "top_p": 0.95},
)
hf_embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5")

Settings.llm = hf_llm
Settings.embed_model = hf_embed_model

# vector_index, summary_index = from_storages()
# vector_tool = get_vector_tool(vector_index=vector_index)
# summary_tool = get_summary_tool(summary_index=summary_index)
# comparison_tool = get_comparision_tool(vector_index=vector_index)
# negotiate_tool = get_negotiation_tool(vector_index=vector_index)

# llm = OpenAI(model="gpt-4o-mini")
# agent = ReActAgent.from_tools(
#     tools=[
#         vector_tool,
#         summary_tool,
#         comparison_tool,
#         negotiate_tool,
#     ],
#     verbose=True,
#     llm=llm,
#     context='This agent assists users with finding apartments, comparing options, and negotiating offers.'
# )

query_engine = router_query_engine()


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
