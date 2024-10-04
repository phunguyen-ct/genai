import os
import subprocess
import openai
import gradio as gr
import phoenix as px

import llama_index.core

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
)
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec

from beautify import beautify
from utils import (
    init_index,
    get_vector_tool,
    get_summary_tool,
    # get_comparision_tool,
    # get_negotiation_tool,
)

from gptcache import cache
# from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# For exponential backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

px.launch_app()
llama_index.core.set_global_handler("arize_phoenix")

# Make sure to: export OPENAI_API_KEY=<Your OpenAI key>
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

# onnx = Onnx()
# data_manager = get_data_manager(
#     CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
# cache.init(
#     embedding_func=onnx.to_embeddings,
#     data_manager=data_manager,
#     similarity_evaluation=SearchDistanceEvaluation(),
# )
# cache.set_openai_key()

# mlflow.llama_index.autolog()  # This is for enabling tracing

Settings.llm = Gemini(model="models/gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding(model="models/embedding-001")


uris = [
    'can-ho-chung-cu-tai-quan-binh-tan',
    'can-ho-chung-cu-tai-quan-thu-duc',
    'can-ho-chung-cu-tai-quan-go-vap',
    'can-ho-chung-cu-tai-quan-tan-phu',
    'can-ho-chung-cu-tai-quan-tan-binh',
    # 'can-ho-chung-cu-tai-quan-phu-nhuan',
    'can-ho-chung-cu-tai-binh-thanh',
    'can-ho-chung-cu-tai-quan-1',
    'can-ho-chung-cu-tai-quan-2',
    'can-ho-chung-cu-tai-quan-3',
    'can-ho-chung-cu-tai-quan-4',
    'can-ho-chung-cu-tai-quan-5',
    'can-ho-chung-cu-tai-quan-6',
    'can-ho-chung-cu-tai-quan-7',
    # 'can-ho-chung-cu-tai-quan-8',
    'can-ho-chung-cu-tai-quan-9',
    # 'can-ho-chung-cu-tai-quan-10',
    # 'can-ho-chung-cu-tai-quan-11',
    # 'can-ho-chung-cu-tai-quan-12'
]

tools_dict = {}
for uri in uris:
    vector_index, summary_index = init_index(name=uri)
    vector_tool = get_vector_tool(vector_index=vector_index, tool_name=uri)
    summary_tool = get_summary_tool(summary_index=summary_index, tool_name=uri)
    tools_dict[uri] = [vector_tool, summary_tool]

# comparison_tool = get_comparision_tool(vector_index=vector_index)
# negotiate_tool = get_negotiation_tool(vector_index=vector_index)

tavily_tool = TavilyToolSpec(
    api_key=os.getenv('TAVILY_API_KEY'),
)

initial_tools = [t for uri in uris for t in tools_dict[uri]] + tavily_tool.to_tool_list()

# gpt-4o-mini
# llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(
    tools=initial_tools,
    verbose=True,
    llm=Settings.llm,
    context='This agent assists users with finding apartments, comparing options, and negotiating offers.'
)


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
    answer = agent.chat(query)

    return beautify(answer)


# Create Gradio Interface
iface = gr.Interface(
    fn=query_with_sources,
    inputs="text",
    outputs=gr.HTML(),
    title="CT",
    description="Thuê căn hộ chung cư ở các quận/huyện Tp HCM"
)

# Launch the interface
iface.launch()

# Start the MLflow UI in a background process
mlflow_ui_command = ["mlflow", "ui", "--port", "5000"]

subprocess.Popen(
    mlflow_ui_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    preexec_fn=os.setsid,
)
