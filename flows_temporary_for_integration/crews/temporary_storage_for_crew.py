from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.llms.groq import Groq
import faiss


faiss_index = faiss.IndexFlatL2(768)

chat_vector_store = FaissVectorStore(faiss_index=faiss_index)

embed_model = HuggingFaceEmbedding(model_name='intfloat/e5-base')


def docs_input_to_nodes(input_message: str = None, output_message: str = None):
    docs = []
    if isinstance(input_message, str):
         docs.append(Document(
            text=input_message,
            metadata={
                'role': 'user',
                'message_type': 'chat_context'
            }
        ))
    if isinstance(output_message, str):
        docs.append(Document(
            text=output_message,
            metadata={
                'role': 'assistant',
                'message_type': 'chat_context'
            }
        ))
    if not docs:
        return None
    return docs

def nodes_to_vector():
    docs = docs_input_to_nodes()
    chunker = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = chunker.get_nodes_from_documents(docs)
    storage_context = StorageContext.from_defaults(vector_store=chat_vector_store)
    storage_context.docstore.add_documents(nodes)
    index = VectorStoreIndex(storage_context=storage_context, embed_model=embed_model)