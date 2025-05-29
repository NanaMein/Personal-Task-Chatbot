# from functools import lru_cache
#
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq as ChatGroq
# from llama_index.vector_stores.milvus import MilvusVectorStore
# from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
# from llama_index.core import VectorStoreIndex
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
#         uri=os.getenv('NEW_URI'),
#         token=os.getenv('NEW_TOKEN'),
#         collection_name='Baby_Fionica_Mock',
#         dim=768,
#         embedding_field='embeddings',
#         enable_sparse=True,
#         enable_dense=True,
#         overwrite=True,
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
