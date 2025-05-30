import os
from dotenv import load_dotenv
import asyncio
from groq import AsyncGroq
from groq.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)

async def intent_router_llm(user_input: str):
    client = AsyncGroq(
        api_key=os.getenv('NEW_API_KEY')
    )
    system_param = ChatCompletionSystemMessageParam(
        role="system",
        content="""
            ### INTENTS: [general_chat], [note_taking], [set_reminder], [memory_context]
            
            ### INSTRUCTIONS: 
            Based on the user input, you need to know the intents, and then answer
            in one of these INTENTS as outputs.
            
            <Examples>
            Use [general_chat] If User is asking for greetings, simple conversation, long and short 
            story telling, question and answer, and general topics.
            User input: Hello there! im david nice to meet you. What's up for lunch?
            Expected output: [general chat]
            
            Use [note_taking] when user is asking about taking notes that user want you
            to keep. Or saving certain information or details
            User input: Take note that im a college student 
            Expected output: [note_taking]
            
            Use [set_reminder] when user input is about to ask to save, retrieve, search, know, update
            and mark finished those reminders.
            User input: Remind me around 5 pm that i am going out
            Expected output: [set_reminder]
            
            Use [memory_context] when user asks about certain past conversation, chat history, query
            certain timeline.
            User input: Do you remember or know my past conversation with you?
            Expected output: [memory_context]
            </Examples>
            
            ### Additional Note: 
            If user input content is not under one of these intents or not safe, please reply
            [not_safe] as output to ensure as a failsafe backup response.
            """
    )
    user_param = ChatCompletionUserMessageParam(
        role="user",
        content=f"### User Input: [({user_input})]"
    )
    chat_completion = await client.chat.completions.create(
        messages=[system_param, user_param],
        model="llama3-8b-8192",
        temperature=.5,
        top_p=.5,
        stop=None,
        stream=False,
    )

    # Print the completion returned by the LLM.
    return chat_completion.choices[0].message.content

# asyncio.run(main())
