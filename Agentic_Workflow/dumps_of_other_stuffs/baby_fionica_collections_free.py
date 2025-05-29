# from functools import lru_cache
#
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq as ChatGroq
# from llama_index.vector_stores.milvus import MilvusVectorStore
# from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
# from llama_index.core import (
#     VectorStoreIndex
# )
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
#
# embed_model = HuggingFaceEmbedding(model_name='intfloat/e5-base')
#
# llm = ChatGroq(
#     model=os.getenv('LLM_SMALL'),
#     api_key=os.getenv('NEW_API_KEY'),
#     temperature=.9,
#     max_new_tokens=3000
# )
#
# @lru_cache(maxsize=1)
# def milvus_vector():
#     vector_store = MilvusVectorStore(
#         uri=os.getenv('uri'),
#         token=os.getenv('token'),
#         collection_name='baby_fionica_collections',
#         dim=768,
#         embedding_field='embeddings',
#         enable_sparse=True,
#         enable_dense=True,
#         overwrite=False,#DONT CHANGE TO TRUE PLEASE
#         sparse_embedding_function=BGEM3SparseEmbeddingFunction()
#     )
#     return VectorStoreIndex.from_vector_store(
#         vector_store=vector_store,
#         embed_model=embed_model
#     )
# index = milvus_vector()
#
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
# async def query_engine_chat_async(inputs: str) -> str:
#     query_engine = index.as_query_engine(
#         llm=llm,
#         vector_store_query_mode="hybrid",
#         similarity_top_k=5
#     )
#     result = query_engine.query(inputs)
#     return result.response
#
#
#
# llm_compound = ChatGroq(
#     model='compound-beta-mini',
#     api_key=os.getenv('NEW_API_KEY'),
#     temperature=.9,
#     max_new_tokens=3000
#
# )
# async def compound_beta_async(inputs: str) -> str:
#     query_engine = index.as_query_engine(
#         llm=llm_compound,
#         vector_store_query_mode="hybrid",
#         similarity_top_k=5
#     )
#     result = query_engine.query(inputs)
#     return result.response
#
#
# async def router_llm(inp_msg: str) -> str:
#     index = VectorStoreIndex().as_chat_engine(
#         llm=llm
#     )
#     inp_msg_temp = f"""
#     ### System: You are a router agent. You will only reply in 3 answers PRIMARY, SECONDARY and FINAL.
#     One of these three will be your output based on context provided
#
#     ### Instructions:
#     You will follow these instructions based on the Input Message
#
#     A: When you think that it falls into the category of simple chatbot and faq, expected output is PRIMARY.
#     B: When you think that its very complex that you need the help of web and learn up to date and real
#     time information outside the context, expected output is SECONDARY
#     C: When you think its not A or B, Primary or Secondary answer, expected output is FINAL.
#
#     ### Input message: [{inp_msg}]
#     """
#     return index.chat(inp_msg_temp).response