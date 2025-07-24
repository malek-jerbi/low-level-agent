import json
import asyncio
import utils
from typing import Any, List, Type, Union
from pydantic import BaseModel
from typing_extensions import Literal
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from dotenv import load_dotenv
from json_parser import from_str


load_dotenv()


Number = Union[int, float]

class Divide(BaseModel):
    reasoning: str
    intent: Literal["divide"] = "divide"
    numerator: Number
    denominator: Number

class Add(BaseModel):
    reasoning: str
    intent: Literal["add"] = "add"
    a: Number
    b: Number

class Multiply(BaseModel):
    reasoning: str
    intent: Literal["multiply"] = "multiply"
    a: Number
    b: Number

class DoneForNow(BaseModel):
    reasoning: str
    intent: Literal["done_for_now"] = "done_for_now"
    message: str


TOOL_MODELS: List[Type[BaseModel]] = [Divide, Add, DoneForNow, Multiply]


SCHEMA_BLOCK = "\nor\n".join(utils.snippet(m) for m in TOOL_MODELS)

class Event:
    def __init__(self, type: str, data: Any):
        self.type = type
        self.data = data

    def as_dict(self) -> dict:
        return {"type": self.type, "data": self.data}

class Thread:
    def __init__(self, events: List[Event]):
        self.events = events

    def serialize_for_llm(self) -> str:
        # change this to XML or your own format if you prefer
        return json.dumps([e.as_dict() for e in self.events], indent=2)

def _build_prompt(thread: Thread) -> str:
    return f"""You are working on the following thread:

{thread.serialize_for_llm()}

What should the next step be?

Answer in JSON using any of these schemas:
{SCHEMA_BLOCK}
"""

thread = Thread([])

async def agent_loop():
    while True:
        prompt = _build_prompt(thread)
        print("The thread that is getting sent to the model in the user prompt literally:\n")
        print("*"*50)
        print(prompt)
        print("*"*50)

        resp = await model_request(
            "openai:gpt-4.1",
            [ModelRequest(parts=[SystemPromptPart(content="You are Asisstant"), UserPromptPart(content=prompt)])],
            model_request_parameters=ModelRequestParameters(output_mode="prompted"),
        )
        
        print("the model's full response response (before we parse and handle it): ")
        print("*"*50)
        print(resp.parts[0].content)
        print("*"*50)

        data = from_str(resp.parts[0].content)

        thread.events.append(Event(type="assistant_action", data=data))
    
        if "intent" in data:
            if data["intent"] == "done_for_now":
                return data["message"]
            
            elif data["intent"] == "divide":
                result = data["numerator"] / data["denominator"]
                thread.events.append(Event(type="tool_result", data={"tool": "divide", "result": result}))

            elif data["intent"] == "add":
                result = data["a"] + data["b"]
                thread.events.append(Event(type="tool_result", data={"tool": "add", "result": result}))

            elif data["intent"] == "multiply":
                result = data["a"] * data["b"]
                thread.events.append(Event(type="tool_result", data={"tool": "multiply", "result": result}))

            else:
                thread.events.append(Event(type="probable_system_error", data={"error_message": "This intent is not handled in the code"}))
                
        else:
            thread.events.append(Event(type="probable_assistant_mistake", data={"error_message": "Unexpected format, missing intent"}))


async def main():
    thread.events.append(Event("start_of_conversation", "This is event 0. your cue to start the conversation with the user using the done_for_now to communicate with the user."))
    result = await agent_loop()
    print(result)

    while True:
        human_query = input()
        thread.events.append(Event("user_input", human_query))
        result = await agent_loop()
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
