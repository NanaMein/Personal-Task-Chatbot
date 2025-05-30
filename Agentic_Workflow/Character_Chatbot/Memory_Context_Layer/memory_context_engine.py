import asyncio

from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as ChatGroq
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction, BM25BuiltInFunction
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document
)
from llama_index.core.text_splitter import SentenceSplitter
from dotenv import load_dotenv
from functools import lru_cache
import os
# Get current time and 1 day ago
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from pymilvus import DataType

load_dotenv()
print("starting embed")
#*******************************************************************************************
embed_model = HuggingFaceEmbedding(model_name='intfloat/e5-base')

llm = ChatGroq(
    model=os.getenv('LLM_SMALL'),
    api_key=os.getenv('NEW_API_KEY'),
    temperature=.3
)
print("MilvusVectorConnection")
@lru_cache(maxsize=1)
def get_vector_store():
    milvus_vector_store = MilvusVectorStore(
        uri=os.getenv('NEW_URI'),
        token=os.getenv('NEW_TOKEN'),
        collection_name='Portfolio_Chat_Context',
        dim=768,
        embedding_field='embeddings',
        enable_sparse=True,
        enable_dense=True,
        overwrite=True,# CHANGE IT FOR DEVELOPMENT STAGE ONLY
        sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
        search_config={"nprobe": 20},
        similarity_metric="L2",  # or "IP"
        consistency_level="Strong",
        hybrid_ranker="WeightedRanker",
        hybrid_ranker_params={"weights": [0.7, 0.3]},
    )
    return milvus_vector_store
    # message_store = [
    #     f"[Role => User// Content => {user_content}]",
    #     f"[Role => Fionica// Content => {assistant_content}]"
    # ]
    # current_timestamp = datetime.now(timezone.utc).isoformat()  # Or use `datetime.now()` for local time
    # current_timestamp = datetime.now(ZoneInfo("Asia/Manila")).isoformat()
    # documents = [Document(
    #     text=text,metadata={
    #             "source": "chat_context",  # Required for Milvus
    #             "timestamp": current_timestamp  # Add timestamp
    #         })
    #     for text in message_store]

    # documents_v1 = [
    #     Document(
    #         text="""
    #
    #             """,
    #         metadata={
    #             "source": "chat_context",
    #             "timestamp": current_timestamp
    #         }
    #     ),
    #     Document(
    #         text="[Role => Fionica// Content => You can find them in the chat history under your profile.]",
    #         metadata={
    #             "source": "chat_context",
    #             "timestamp": current_timestamp
    #         }
    #     )
    # ]
    #
    # parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    #
    # nodes = parser.get_nodes_from_documents(documents=documents)
    #
    # index = VectorStoreIndex(
    #     nodes=nodes,
    #     vector_store=lazy_vector,
    #     embed_model=embed_model,
    #     storage_context=StorageContext.from_defaults(
    #         vector_store=lazy_vector
    #     )
    # )


def lazy_loader():

    return get_vector_store()

print("lazy loader on the roll")
lazy_vector = lazy_loader()

async def indexed_chat_context(user_content:str , assistant_content:str, intents:str):
    current_timestamp = datetime.now(ZoneInfo("Asia/Manila")).isoformat()

    documents_v1 = [
        Document(
            text=f"""
                [Role: User\'s query
                //
                Content: {user_content}]
                """,
            metadata={
                "source": "chat_context",
                "timestamp": current_timestamp,
                "role" : "user",
                "location" : "philippines",
                "intent_tags":
                    [intents]
            }
        ),
        Document(
            text=f"""
                [Role: assistant\'s response
                //
                Content: {assistant_content}]
                """,
            metadata={
                "source": "chat_context",
                "timestamp": current_timestamp,
                "role": "assistant",
                "location": "philippines",
                "intent_tags":
                    [intents]
            }
        )
    ]

    # message_store = [
    #     f"[Role => User// Content => {user_content}]",
    #     f"[Role => Fionica// Content => {assistant_content}]"
    # ]
    # current_timestamp = datetime.now(timezone.utc).isoformat()  # Or use `datetime.now()` for local time
    # current_timestamp = datetime.now(ZoneInfo("Asia/Manila")).isoformat()
    # documents = [Document(
    #     text=text,metadata={
    #             "source": "chat_context",  # Required for Milvus
    #             "timestamp": current_timestamp  # Add timestamp
    #         })
    #     for text in message_store]

    parser = SentenceSplitter(chunk_size=400, chunk_overlap=40)

    nodes = parser.get_nodes_from_documents(documents=documents_v1)

    index = VectorStoreIndex(
        nodes=nodes,
        vector_store=lazy_vector,
        embed_model=embed_model,
        storage_context=StorageContext.from_defaults(
            vector_store=lazy_vector
        )
    )
async def indexed_query_engine(input_question: str) -> str:

    index = VectorStoreIndex.from_vector_store(
        vector_store=lazy_vector,
        embed_model=embed_model
    )
    query_engine = index.as_query_engine(
        llm=llm,
        vector_store_query_mode="hybrid",
        similarity_top_k=5
    )
    obj_str = query_engine.query(input_question)
    return obj_str.response

# def query_engine_chat(inputs: str) -> str:
#     return str(query_engine.query(inputs))

# async def memory_query_engine(prompt: str) -> str:
#     prompt_template = f"""
#         ### System: You are a Chat Context Storage and you are a collection of past conversations.
#         ### Input Query: [{prompt}]
#         ### Instructions: There are only 2 possible answer you will reply based on Input Query.
#         When input query ask for context and it exist, expected output is the answer to that context.
#         When input Query cant find the context or doesnt exit, expected output is NO MEMORY STORED.
#         ### Expected Output: Retrieve only relevant, similar or same information based on Input Query,
#         dont generate answer, Just retrieval only. If no information retrieval happen, expected output is
#         NO MEMORY STORED.
#         """
#     context = await indexed_query_engine(prompt_template)
#     context_template = f"""
#         ### User Current Input Query: [{prompt}]
#         ### User Previous Context: [{context}]
#         ### Instruction: Use the User Previous Context as reference only, Also if the current query
#         is not related or asks about the previous context, ignore user previous context. Use it only
#         as reference and when it is only asked.
#         """
#     # context_template_output = await indexed_query_engine(context_template)
#     # await indexed_chat_context(prompt, context_template_output)
#     # return context_template_output
#     return context_template
#
# print("starting loop")
# async def memory_run_async():
#     while True:
#         print("testing\n")
#         question = input("Write something you like (type 'exit' to quit): \n\n")
#
#         if question.lower() == "exit":
#             print("Exiting the loop.")
#             break
#
#         # output = await indexed_query_engine(question)
#         # await indexed_chat_context(question,output)
#         output = await memory_query_engine(question)
#         print(output)
#         print("****************************************************************\n\n")
#
# asyncio.run(memory_run_async())








