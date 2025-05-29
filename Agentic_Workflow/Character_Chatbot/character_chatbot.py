import asyncio

# from Agentic_Workflow.Character_Chatbot.Memory_Context_Layer.memory_context_engine import indexed_query_engine
from .Memory_Context_Layer.memory_context_engine import indexed_chat_context, indexed_query_engine
from .Personality_Context_Layer.personality_rag_engine import query_engine_chat_async , compound_beta_async, router_llm_async

async def memory_query_engine(prompt: str, web_search: bool = False) -> str:
    """ Combination of Memory Layer and Personality Layer of Fionica and Chat Context"""
    prompt_template = f"""
        ### System: You are a Chat Context Storage and you are a collection of past conversations. 
        ### Input Query: [{prompt}]
        ### Instructions: There are only 2 possible answer you will reply based on Input Query.
        When input query ask for context and it exist, expected output is the answer to that context.
        When input Query cant find the context or doesnt exit, expected output is NO MEMORY STORED.
        ### Expected Output: Retrieve only relevant, similar or same information based on Input Query, 
        dont generate answer, Just retrieval only. If no information retrieval happen, expected output is
        NO MEMORY STORED.
        """

    context = await indexed_query_engine(prompt_template)

    context_template = f"""
        ### User Current Input Query: [{prompt}]
        ### User Previous Context: [{context}]
        ### Instruction: Use the User Previous Context as reference only, Also if the current query
        is not related or asks about the previous context, ignore user previous context. Use it only 
        as reference and when it is only asked.
        """
    context_template_output = await indexed_query_engine(context_template)

    fionica_template = f"""
        
        ### Raw Query Input: [{prompt}]
        ### Generated Response Output: [{context_template_output}]
        ### Instruction: Convert or change the Generated Response Output into something 
        how Fionica, the virtual daughter, will should reply.
        ### Role: [Fionica]
        ### System: [You will roleplay as Fionica] 
        
        """
    if web_search:
        fionica_template_output = await compound_beta_async(fionica_template)
        await indexed_chat_context(prompt, fionica_template_output)
        return fionica_template_output


    fionica_template_output = await query_engine_chat_async(fionica_template)
    await indexed_chat_context(prompt, fionica_template_output)
    return fionica_template_output
    # return context_template


async def fionica_virtual_assistant(input: str, web_search: bool = False) -> str:
    obj = await memory_query_engine(input, web_search)
    return obj