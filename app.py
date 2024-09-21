import os
import subprocess
import mlflow
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

from beautify import beautify
from utils import (
    init_index,
    get_vector_tool,
    get_summary_tool,
    get_comparision_tool,
    get_negotiation_tool,
)

# For exponential backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

px.launch_app()
llama_index.core.set_global_handler("arize_phoenix")

mlflow.llama_index.autolog()  # This is for enabling tracing

# Make sure to: export OPENAI_API_KEY=<Your OpenAI key>
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

Settings.llm = Gemini(model="models/gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding(model="models/embedding-001")

vector_index, summary_index = init_index()

vector_tool = get_vector_tool(vector_index=vector_index)
summary_tool = get_summary_tool(summary_index=summary_index)
comparison_tool = get_comparision_tool(vector_index=vector_index)
negotiate_tool = get_negotiation_tool(vector_index=vector_index)

# llm = OpenAI(model="gpt-3.5-turbo")
agent = ReActAgent.from_tools(
    tools=[
        vector_tool,
        summary_tool,
        comparison_tool,
        negotiate_tool,
    ],
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
    title="Chatbot tư vấn cho thuê căn hộ chung cư",
    description="Chatbot tư vấn cho thuê căn hộ chung cư ở các quận/huyện Tp HCM"
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
