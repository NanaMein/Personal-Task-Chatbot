import os
from typing import Type, Any
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.llms.groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api = os.getenv('NEW_API_KEY')
os.environ["GROQ_API_KEY"] = groq_api
from flows.tools import persona_tool
class MioIkariCrew:
    def __init__(self):
        # self.milvus = self.milvus_sync()
        self.llm = self.llm_def()
        self.tool = persona_tool
        self._chat_llm = self.chat_llm()
        self.fake = self.fake_func_calling()

    def agent(self):
        return Agent(
            role="Roleplayer that is flexible and can take any roles in different scenario and situations",
            backstory="""
                You were an impromptu stage artist. You can impersonate anyone through personality,
                tonality of voice, description, role, traits, intents of any character. You are
                able to get upto 90% to 100% accuracy of the character you are pertaining, acting and
                roleplaying. With little information, you can be flexible and adaptable to anything.
            """,

            goal="""
                Right now you are impersonating an ai assistant, named {virtual_persona}. You will 
                use the tool provided to you to look up for information regarding the character
                profile of {virtual_persona}. You are expected to reply the user query => [\'{input_message}\']
                
            """,
            llm=self.llm,
            tools=[self.tool],
            max_reasoning_attempts=3,
            reasoning=True,
            verbose=True,
            max_retry_limit=5,
            max_iter=5, function_calling_llm=self.fake
            # max_execution_time=30
        )
    def task(self):
        return Task(
            agent=self.agent(),
            description="""
                User Query: [\'{input_message}\']
                Chat History: [\'{chat_history}\']
                Your task is to impersonate and roleplay the character {virtual_persona}. You will most likely do 
                conversational turns with the user. You reply greetings with greetings, with query, you reply with
                your knowledge base and information regarding that. You will reply based on the user query provided to 
                you. Chat history is for reference only and for contextual awareness
                
            """,
            expected_output="""You will roleplay as a character named {virtual_persona} and with the user query 
                => [\'{input_message}\'], You will answer based on that. Use the chat history for contextual awareness
                in order for you not to get lost or hallucination.
            """,

            # async_execution=True
        )

    def llm_def(self):
        return LLM(
            model=os.getenv('llm_small'),
            # api_key=os.getenv('NEW_API_KEY'),
            api_base=os.getenv('API_BASE_GROQ'),
            top_p=.4,
            temperature=.7
        )
    def fake_func_calling(self):
        return LLM(#qwen-qwq-32b #gemma2-9b-it #compound-beta-mini
            model='gemma2-9b-it',
            api_base=os.getenv('API_BASE_GROQ'),
            api_key=os.getenv('NEW_API_KEY')
        )

    def chat_llm(self):
        return LLM(
            model='qwen-qwq-32b',
            api_key=os.getenv('NEW_API_KEY'),
            api_base = os.getenv('API_BASE_GROQ'),
        )

    def crew(self):
        return Crew(
            agents=[self.agent()],
            tasks=[self.task()],
            process=Process.sequential,
            verbose=True, #chat_llm=self._chat_llm
            # function_calling_llm=self._chat_llm
        )




    def kickoff_crew(self, input_message: str, chat_history: str):
        async_crew = self.crew()
        messages ={
            'virtual_persona': "Fionica",
            'input_message': input_message,
            'chat_history':chat_history
        }
        # output = await async_crew.kickoff_async(inputs=messages)
        output = async_crew.kickoff(inputs=messages)
        return output.raw
#
#
# from pydantic import PrivateAttr
# from flows.vector_stores import milvus_connection
#
# class MyToolInput(BaseModel):
#     """Input schema for MyCustomTool."""
#     argument: str = Field(..., description="Used to input questions regarding the character you need to look up")
#
# class MyCustomTool(BaseTool):
#
#
#     name: str = "Character Profile Tool"
#     description: str = """ this tool is a query engine chatbot. It basically stores all details of the character
#         you need. From personality, to behavior, to traits and character background. """
#     args_schema: Type[BaseModel] = MyToolInput
#     _milvus: MilvusVectorStore = PrivateAttr()  # 👈 Private field
#     _embed_model: HuggingFaceEmbedding = PrivateAttr()
#
#     def __init__(self, /, **data: Any):
#         super().__init__(**data)
#         self._milvus = milvus_connection
#         self._embed_model = embed_model
#
#
#     def _run(self, argument: str) -> str:
#         storage_context = StorageContext.from_defaults(vector_store=self._milvus)
#         index = VectorStoreIndex.from_vector_store(
#             vector_store=self._milvus,
#             embed_model=self._embed_model,
#             storage_context=storage_context
#         )
#         llm_groq = Groq(
#             model="llama-3.1-8b-instant",
#             api_key=os.getenv('NEW_API_KEY')
#         )
#         query_engine = index.as_query_engine(llm=llm_groq)
#
#         answer = query_engine.query(argument)
#
#         return answer.response
#
# embed_model = HuggingFaceEmbedding(model_name='intfloat/e5-base')