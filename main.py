# from flask import Flask, request
import os
# import subprocess
import gradio as gr

import llama_index.core

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.core import (
    Settings,
)
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec

# from package_name.module_name import function_name
from packages.Beautify.beautify import beautify
from packages.Utils.utils import (
    init_index,
    get_vector_tool,
    get_summary_tool,
    get_comparision_tool,
    get_negotiation_tool,
)

from packages.Prompt.prompt import react_system_prompt

# For exponential backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# app = Flask(__name__)


os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

# Settings.llm = Gemini(model="models/gemini-1.5-flash-001")
Settings.llm = Ollama(model="llama3.2:latest", request_timeout=120.0)
Settings.embed_model = GeminiEmbedding(model="models/embedding-001")


uris = [
    'can-ho-chung-cu-tai-quan-binh-tan',
    'can-ho-chung-cu-tai-quan-thu-duc',
    'can-ho-chung-cu-tai-quan-go-vap',
    'can-ho-chung-cu-tai-quan-tan-phu',
    'can-ho-chung-cu-tai-quan-tan-binh',
    'can-ho-chung-cu-tai-binh-thanh',
    'can-ho-chung-cu-tai-quan-1',
    'can-ho-chung-cu-tai-quan-2',
    'can-ho-chung-cu-tai-quan-3',
    'can-ho-chung-cu-tai-quan-4',
    'can-ho-chung-cu-tai-quan-5',
    'can-ho-chung-cu-tai-quan-6',
    'can-ho-chung-cu-tai-quan-7',
    'can-ho-chung-cu-tai-quan-9',
]

tools_dict = {}
for uri in uris:
    vector_index, summary_index = init_index(name=uri)
    
    vector_tool = get_vector_tool(vector_index=vector_index, tool_name=uri)
    summary_tool = get_summary_tool(summary_index=summary_index, tool_name=uri)
    comparison_tool = get_comparision_tool(vector_index=vector_index, tool_name=uri)
    negotiate_tool = get_negotiation_tool(vector_index=vector_index, tool_name=uri)
    
    tools_dict[uri] = [vector_tool, summary_tool, comparison_tool, negotiate_tool]


# tavily_tool = TavilyToolSpec(
#     api_key=os.getenv('TAVILY_API_KEY'),
# )

initial_tools = [t for uri in uris for t in tools_dict[uri]]
# initial_tools = [t for uri in uris for t in tools_dict[uri]] + \
#     tavily_tool.to_tool_list()

agent = ReActAgent.from_tools(
    tools=initial_tools,
    verbose=True,
    llm=Settings.llm,
    context='This agent assists users with finding apartments, comparing options, and negotiating offers.'
)
# agent.get_prompts()
# agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
# agent.reset()


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
    title="CT Chatbot",
    theme=gr.themes.Soft(),
    description="Thuê căn hộ chung cư ở các quận/huyện Tp HCM"
)

# Launch the interface
iface.launch()


# @app.route('/')
# def search_response():
#     query = request.args.get("q")

#     if not query:
#         return {"error_message": "Input parameter missing"}, 422

#     return ("My first Flask application in action!", 200)


# @app.errorhandler(500)
# def server_error(error):
#     return {"message": "Something went wrong on the server"}, 500
