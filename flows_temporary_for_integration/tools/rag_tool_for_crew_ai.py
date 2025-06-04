

from typing import Type, Any
from llama_index.llms.groq import Groq
from crewai.tools import BaseTool
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pydantic import PrivateAttr, BaseModel, Field
from flows.vector_stores import milvus_connection
from dotenv import load_dotenv
import os

load_dotenv()



class MyToolInput(BaseModel):
    """Input schema for MyPersonaTool."""
    argument: str = Field(..., description="Used to input questions regarding the character you need to look up")

class MyPersonaTool(BaseTool):


    name: str = "Character Profile Tool"
    description: str = """ this tool is a query engine chatbot. It basically stores all details of the character
        you need. From personality, to behavior, to traits and character background. """
    args_schema: Type[BaseModel] = MyToolInput
    _milvus: MilvusVectorStore = PrivateAttr()
    _embed_model: HuggingFaceEmbedding = PrivateAttr()

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        self._milvus = milvus_connection
        self._embed_model = embed_model


    def _run(self, argument: str) -> str:
        storage_context = StorageContext.from_defaults(vector_store=self._milvus)
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._milvus,
            embed_model=self._embed_model,
            storage_context=storage_context
        )
        llm_groq = Groq(
            # model="llama-3.1-8b-instant",
            model=os.getenv('LLM_SMALL'),
            api_key=os.getenv('NEW_API_KEY')
        )
        query_engine = index.as_query_engine(llm=llm_groq)

        answer = query_engine.query(argument)

        return answer.response

embed_model = HuggingFaceEmbedding(model_name='intfloat/e5-base')