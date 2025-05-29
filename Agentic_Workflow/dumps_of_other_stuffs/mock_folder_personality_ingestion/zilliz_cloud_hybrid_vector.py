print("HYBRID INGESTION LOADING")
from functools import lru_cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as ChatGroq
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os

load_dotenv()

embed_model = HuggingFaceEmbedding(
    model_name='intfloat/multilingual-e5-large',
    # model_kwargs={"device": "cuda"},
    # encode_kwargs={"batch_size": 64}
)

llm = ChatGroq(
    model=os.getenv('LLM_SMALL'),
    api_key=os.getenv('NEW_API_KEY'),
    temperature=.9,
    max_new_tokens=3000
)

@lru_cache(maxsize=1)
def milvus_vector():
    return MilvusVectorStore(
        uri=os.getenv('NEW_URI'),
        token=os.getenv('NEW_TOKEN'),
        collection_name='Baby_Fionica_Mock',
        dim=1024,
        embedding_field='embeddings',
        enable_sparse=True,
        enable_dense=True,
        overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
        sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
        search_config={"nprobe": 20},
        similarity_metric="L2",  # or "IP"
        consistency_level="Strong",
        hybrid_ranker="WeightedRanker",
        hybrid_ranker_params={"weights": [0.7, 0.3]},
    )


vector_milvus = milvus_vector()
# load_documents = SimpleDirectoryReader(input_dir='./data_sources').load_data()

# load_documents = SimpleDirectoryReader(input_files=['personality_source/fionica.yaml']).load_data()
#
# parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
#
# nodes = parser.get_nodes_from_documents(load_documents)

# storage_context = StorageContext.from_defaults(vector_store=vector_milvus)

index = VectorStoreIndex.from_vector_store(
        vector_store=vector_milvus,
        embed_model=embed_model
)

def query_engine_chat(inputs: str) -> str:
    query_engine = index.as_query_engine(
        llm=llm,
        vector_store_query_mode="hybrid",
        similarity_top_k=5
    )
    result = query_engine.query(inputs)
    return result.response

async def query_engine_chat_async(inputs: str) -> str:
    query_engine = index.as_query_engine(
        llm=llm,
        vector_store_query_mode="hybrid",
        similarity_top_k=5
    )
    result = query_engine.query(inputs)
    return result.response


"*************"
"*****************"
"****************"
"***************"
"***************"
"*****"
# load_documents = SimpleDirectoryReader(input_dir='./data_sources').load_data()
#
# parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
#
# nodes = parser.get_nodes_from_documents(load_documents)
#
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
#
# index = VectorStoreIndex(
#     nodes,
#     vector_store=vector_store,
#     embed_model=embed_model,
#     storage_context=storage_context
# )

# query_engine = index.as_query_engine(
#     llm=llm,
#     vector_store_query_mode="hybrid",
#     similarity_top_k=5
# )
#
# def query_engine_chat(inputs: str) -> str:
#     return str(query_engine.query(inputs))

#
# def query_engine_chat(inputs: str) -> str:
#     query_engine = index.as_query_engine(
#         llm=llm,
#         vector_store_query_mode="hybrid",
#         similarity_top_k=5
#     )
#     result = query_engine.query(inputs)
#     return result.response
#
#
# async def query_engine_chat_async(inputs: str) -> str:
#     query_engine = index.as_query_engine(
#         llm=llm,
#         vector_store_query_mode="hybrid",
#         similarity_top_k=5
#     )
#     result = query_engine.query(inputs)
#     return result.response


llm_compound = ChatGroq(
    model='compound-beta-mini',
    api_key=os.getenv('NEW_API_KEY'),
    temperature=.9,
    max_new_tokens=3000
)


async def compound_beta_async(inputs: str) -> str:
    query_engine = index.as_query_engine(
        llm=llm_compound,
        vector_store_query_mode="hybrid",
        similarity_top_k=5
    )
    result = query_engine.query(inputs)
    return result.response


async def router_llm_async(inp_msg: str) -> str:
    index = VectorStoreIndex().as_chat_engine(
        llm=llm
    )
    inp_msg_temp = f"""
    ### System: You are a router agent. You will only reply in 3 answers PRIMARY, SECONDARY and FINAL.
    One of these three will be your output based on context provided

    ### Instructions: 
    You will follow these instructions based on the Input Message

    A: When you think that it falls into the category of simple chatbot and faq, expected output is PRIMARY.
    B: When you think that its very complex that you need the help of web and learn up to date and real
    time information outside the context, expected output is SECONDARY
    C: When you think its not A or B, Primary or Secondary answer, expected output is FINAL.

    ### Input message: [{inp_msg}]
    """
    return index.chat(inp_msg_temp).response

print("HYBRID INGESTION COMPLETE")
