from datetime import datetime
from zoneinfo import ZoneInfo

print("ASYNC FLOW LOADING")

from crewai.flow import Flow, start, listen, router
from pydantic import BaseModel
from dotenv import load_dotenv

# from .Character_Chatbot.Personality_Context_Layer.personality_rag_engine import router_llm_async
from .Character_Chatbot.personal_task_chatbot import fionica_virtual_assistant
from .Character_Chatbot.Router_Context_Layer.router_logic_engine import intent_router_llm

load_dotenv()

class FlowState(BaseModel):
    input: str = ""
    router_choices: str = ""
    router_output: str = ""
    fake_router_output: str=""




class PersonalTaskFlow(Flow[FlowState]):



    @start()
    async def memory_context_flow(self):
        current_timestamp = datetime.now(ZoneInfo("Asia/Manila")).isoformat()


    @listen(memory_context_flow)
    async def start_method(self):
        router_llm = await intent_router_llm(self.state.input)
        self.state.router_output = router_llm.strip()

    @router(start_method)
    async def routing_llm(self):
        routing_obj = self.state.router_output
        if routing_obj=='[general_chat]':
            return "general_chat"

        elif routing_obj=='[note_taking]':
            return "note_taking"
        elif routing_obj=='[set_reminder]':
            return "set_reminder"
        elif routing_obj=='[memory_context]':
            return "memory_context"

        else:
            return "FINAL"


    @listen("PRIMARY")
    async def choice_1(self):
        primary = await fionica_virtual_assistant(self.state.input)

        return f"""
        User Input: {self.state.input}\n
        Assistant Output: {primary}\n
        Test router intent: {self.state.fake_router_output}
        """

    @listen("SECONDARY")
    async def choice_2(self):
        secondary = await fionica_virtual_assistant(self.state.input, web_search=True)
        return f"""
               User Input: {self.state.input}\n
               Assistant Output: {secondary}\n
               Test router intent: {self.state.fake_router_output}
               """

    @listen("FINAL")
    async def choice_3(self):

        return f"Error Output: **{self.state.router_output}**\n router choice {self.state.fake_router_output}"


async def personal_task_wrapper(input_message: str) -> str:
    obj = PersonalTaskFlow()
    # obj.plot("my_flow_schema_structure")
    async_obj = await obj.kickoff_async(inputs={'input': input_message})
    return str(async_obj)


# test = asyncio.run(personal_task_wrapper())
# print(test)

print("ASYNC FLOW COMPLETE")
