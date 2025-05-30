from datetime import datetime
from zoneinfo import ZoneInfo

print("ASYNC FLOW LOADING")

from crewai.flow import Flow, start, listen, router
from pydantic import BaseModel
from dotenv import load_dotenv
#
# from .Character_Chatbot.Personality_Context_Layer.personality_rag_engine import router_llm_async
# from .Character_Chatbot.personal_task_chatbot import fionica_virtual_assistant

load_dotenv()

class FlowState(BaseModel):
    input: str = ""
    router_choices: str = ""
    router_output: str = ""
    history_context: str = ""
    fake_router_output: str=""




class PersonalTaskFlow(Flow[FlowState]):



    @start()
    async def memory_context_flow(self):
        current_timestamp = datetime.now(ZoneInfo("Asia/Manila")).isoformat()


    @listen(memory_context_flow)
    async def start_method(self):
        router_llm = await router_llm_async(self.state.input)
        self.state.router_output = router_llm.strip()

    @router(start_method)
    async def routing_llm(self):
        routing_obj = self.state.router_output
        if routing_obj=='PRIMARY':
            return "PRIMARY"

        elif routing_obj=='SECONDARY':
            return "SECONDARY"

        else:
            return "FINAL"


    @listen("PRIMARY")
    async def choice_1(self):
        return primary

    @listen("SECONDARY")
    async def choice_2(self):
        return secondary

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
