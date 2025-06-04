# from crews.mio_ikari_crew import MioIkariCrew
from flows.crews.mio_ikari_crew import MioIkariCrew
# from flows import MioIkariCrew
from crewai.flow import Flow, start, listen, router, or_, and_, persist
from pydantic import BaseModel
from collections import deque
from dotenv import load_dotenv
import os

load_dotenv()
persist_history = deque(maxlen=8)
persist_history.append("*[N/A]")

class FlowState(BaseModel):
    entry_point: str = ""
    mid_point: str = ""
    # chat_history: str = ""

class PersonalTaskFlow(Flow[FlowState]):
    def __init__(self):
        super().__init__()
        self.flow = MioIkariCrew()

    @start()
    def start(self):
        self.state.mid_point = self.flow.kickoff_crew(
            self.state.entry_point,
            '\n--\n'.join(persist_history)
        )

    # @listen(start)
    # def next(self):
    #     pass

    @listen(start)
    def intent_identifier(self):
        input_message = self.state.entry_point
        output_message = self.state.mid_point
        persist_history.append(f"|[role=user]*[content=\'{input_message}\']|")
        persist_history.append(f"|[role=assistant]*[content=\'{output_message}\']|")
        return output_message

    # @router(intent_identifier)
    # def message_category(self):
    #     pass



