print("HYBRID INGESTION LOADING")
from functools import lru_cache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as ChatGroq
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os

load_dotenv()

embed_model = HuggingFaceEmbedding(model_name='intfloat/e5-base')

def gemma():
    return ChatGroq(
    model = "gemma2-9b-it",
    api_key = os.getenv('NEW_API_KEY'),
    temperature = .3,
    )

def llm():
    return ChatGroq(
        model=os.getenv('LLM_SMALL'),
        api_key=os.getenv('NEW_API_KEY'),
        temperature=.9,
        max_new_tokens=3000
    )


@lru_cache(maxsize=1)
def index():
    vector = MilvusVectorStore(
        uri=os.getenv('NEW_URI'),
        token=os.getenv('NEW_TOKEN'),
        collection_name='Baby_Fionica_Mock',
        dim=768,
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

    return VectorStoreIndex.from_vector_store(
            vector_store=vector,
            embed_model=embed_model
    )
index = index()

# def query_engine_chat(inputs: str) -> str:
#     query_engine = index.as_query_engine(
#         llm=llm(),
#         vector_store_query_mode="hybrid",
#         similarity_top_k=5
#     )
#     result = query_engine.query(inputs)
#     return result.response

async def query_engine_chat_async(inputs: str) -> str:
    query_engine = index.as_query_engine(
        llm=llm(),
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
"Your mom angelica tumbaga tremor works in cavite philippines, do you think in june 6, 2025 is a holiday for her? so she could take a break"




# async def router_llm_async(inp_msg: str) -> str:
#     query_index = VectorStoreIndex().as_chat_engine(
#         llm=llm()
#     )


async def router_llm_async(inp_msg: str) -> str:
    query_index = index.as_query_engine(
        llm=gemma()
    )
    inp_msg_temp = f"""
    ### System: You are a router agent. You will only reply in 3 answers PRIMARY, SECONDARY and FINAL.
    One of these three will be your output based on context provided
    
    ### Input message: [{inp_msg}]
    
    ### Instructions: 
    You will follow these instructions based on the Input Message. 
    
    Note to Tagalog Language: Dont put tagalog to FINAL output unless
    you think that it cant be under the category of PRIMARY or SECONDARY. 

    A: When you think that it falls into the category of simple chatbot and faq, expected output is PRIMARY.
    B: When you think that its very complex that you need the help of web and learn up to date and real
    time information outside the context, expected output is SECONDARY
    C: FINAL is for very complexity or very nonsense and no common sense, Unless its something you dont understand
    dont put it to final, but if its something hard to understand. Use FINAL as expected output

    
    """
    return query_index.query(inp_msg_temp).response

print("HYBRID INGESTION COMPLETE")
