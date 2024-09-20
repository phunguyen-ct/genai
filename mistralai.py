import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import (
    Settings,
    PromptTemplate,
)

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