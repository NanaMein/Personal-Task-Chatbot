import asyncio

print("HYBRID INGESTION LOADING")
from functools import lru_cache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as ChatGroq
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from dotenv import load_dotenv
import os

load_dotenv()

embed_model = HuggingFaceEmbedding(model_name='intfloat/e5-base')



@lru_cache(maxsize=1)
def milvus_vector():
    return MilvusVectorStore(
        uri=os.getenv('NEW_URI'),
        token=os.getenv('NEW_TOKEN'),
        collection_name='Baby_Fionica_Mock',
        dim=768,
        embedding_field='embeddings',
        enable_sparse=True,
        enable_dense=True,
        overwrite=True,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
        sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
        search_config={"nprobe": 20},
        similarity_metric="L2",  # or "IP"
        consistency_level="Strong",
        hybrid_ranker="WeightedRanker",
        hybrid_ranker_params={"weights": [0.7, 0.3]},
    )


vector_milvus = milvus_vector()

load_documents = SimpleDirectoryReader(input_files=['personality_source/fionica.yaml']).load_data()

parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

nodes = parser.get_nodes_from_documents(load_documents)

storage_context = StorageContext.from_defaults(vector_store=vector_milvus)

index = VectorStoreIndex(
        nodes=nodes,
        vector_store=vector_milvus,
        embed_model=embed_model,
        storage_context=storage_context
)

llm = ChatGroq(
    model=os.getenv('LLM_SMALL'),
    # model="groq/qwen-qwq-32b",
    api_key=os.getenv('NEW_API_KEY'),
    temperature=.7,
    max_new_tokens=5000
)

llm_router = ChatGroq(
    # model=os.getenv('LLM_BIG'),
    model=os.getenv('LLM_SMALL'),
    api_key=os.getenv('NEW_API_KEY'),
    temperature=.7,
    max_new_tokens=5000
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

llm_compound = ChatGroq(
    model='compound-beta-mini',
    api_key=os.getenv('NEW_API_KEY'),
    temperature=.5
    # max_new_tokens=3000
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
    index_query = index.as_query_engine(llm=llm_router)

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
    
    ### Expected Output: A, B or C as an output only.
    """
    return index_query.query(inp_msg_temp).response

async def test_wrapper_async(input: str):
    # input = f"""
    #         ### SYSTEM: You are Fionica, the virtual assistant.
    #         ### USER INPUT: [{input}]
    #         ### Expected output: You will reply USER INPUT based on
    #         context and information you have.
    #         """

    test1 = await router_llm_async(input)
    test2 = await query_engine_chat_async(input)
    test3 = await compound_beta_async(input)
    result = (f"Router: {test1}\n"
                +f"USER: {input}\n"
                +f"CHATBOT:\t {test2}\n"
                +f"WEB SEARCH\t: {test3}\n")
    return result


# print("HYBRID INGESTION COMPLETE")
# while True:
#     print("test")
#     messages = input(" write something to me please: ")
#     if messages=="exit":
#         break
#     print(asyncio.run(test_wrapper_async(messages)))

"Your mom angelica works in cavite philippines, do you think in june 6, 2025 is a holiday for her? so she could take a break"