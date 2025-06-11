from langgraph.graph import StateGraph # type: ignore
from pydantic import BaseModel # type: ignore
from typing import Annotated, Sequence, Literal, TypedDict
from operator import add
from langchain_core.messages import BaseMessage # type: ignore
from langchain_core.output_parsers import PydanticOutputParser # type: ignore

class SupervisorOutputParser(BaseModel):
    node_selection_type: Literal['rag', 'llm', 'crawler']

parser = PydanticOutputParser(pydantic_object = SupervisorOutputParser)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]

format_instructions = parser.get_format_instructions()