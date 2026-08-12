import os
import json
import uuid
import hashlib

from datetime import datetime, timedelta
from typing import TypedDict, List, Optional, Annotated

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langserve import add_routes

import chromadb


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# ============================================================
# 2. GEMINI MODEL
# ============================================================

# IMPORTANT:
# gemini-3.1-flash-lite-preview was shut down.
# Use the stable model.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. FILE LOCATIONS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SCHEDULE_FILE = os.path.join(
    BASE_DIR,
    "schedule.json"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# 4. STATE
# ============================================================

class ScheduleState(TypedDict):
    messages: Annotated[
        List[BaseMessage],
        add_messages
    ]

    user_query: str

    tool_result: Optional[str]

    final_answer: Optional[str]


# ============================================================
# 5. CREATE SAMPLE SCHEDULE
# ============================================================

def create_sample_schedule():

    today = datetime.now().date()

    schedule = []

    def add_event(
        event_id,
        title,
        event_type,
        days_from_today,
        start_time,
        end_time,
        location,
        description
    ):

        event_date = (
            today +
            timedelta(days=days_from_today)
        ).isoformat()

        schedule.append({
            "id": event_id,
            "title": title,
            "type": event_type,
            "date": event_date,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
            "description": description
        })

    add_event(
        "event-001",
        "Team Standup",
        "meeting",
        0,
        "10:00",
        "10:30",
        "Online",
        "Daily project status meeting."
    )

    add_event(
        "event-002",
        "DSA Practice",
        "task",
        1,
        "09:00",
        "10:00",
        "Hostel",
        "Practice arrays and sliding window."
    )

    add_event(
        "event-003",
        "AI Workshop",
        "workshop",
        2,
        "14:00",
        "17:00",
        "College Lab",
        "Workshop about Agentic AI and RAG."
    )

    add_event(
        "event-004",
        "Project Meeting",
        "meeting",
        3,
        "14:00",
        "15:00",
        "Online",
        "Discuss major project progress."
    )

    add_event(
        "event-005",
        "Doctor Appointment",
        "appointment",
        5,
        "11:00",
        "12:00",
        "City Clinic",
        "Regular appointment."
    )

    add_event(
        "event-006",
        "Java Practice",
        "task",
        7,
        "18:00",
        "19:30",
        "Hostel",
        "Practice Java collections."
    )

    add_event(
        "event-007",
        "Cloud Workshop",
        "workshop",
        10,
        "10:00",
        "13:00",
        "College",
        "Cloud deployment workshop."
    )

    add_event(
        "event-008",
        "Project Review",
        "meeting",
        14,
        "15:00",
        "16:00",
        "Online",
        "Review Agentic RAG project."
    )

    add_event(
        "event-009",
        "Assignment Submission",
        "task",
        20,
        "12:00",
        "13:00",
        "College Portal",
        "Submit AI project assignment."
    )

    add_event(
        "event-010",
        "Career Workshop",
        "workshop",
        25,
        "16:00",
        "18:00",
        "Seminar Hall",
        "Resume and interview workshop."
    )

    return schedule


# ============================================================
# 6. LOAD SCHEDULE
# ============================================================

def load_schedule():

    if not os.path.exists(SCHEDULE_FILE):

        schedule = create_sample_schedule()

        with open(
            SCHEDULE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                schedule,
                file,
                indent=4
            )

        return schedule

    try:

        with open(
            SCHEDULE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except Exception as error:

        print(
            "Schedule loading error:",
            error
        )

        return []


# ============================================================
# 7. SAVE SCHEDULE
# ============================================================

def save_schedule(schedule):

    with open(
        SCHEDULE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            schedule,
            file,
            indent=4
        )


# ============================================================
# 8. SIMPLE LOCAL VECTOR EMBEDDING
# ============================================================

# We use a deterministic hashing-based embedding.
#
# Why?
#
# Chroma's default embedding model can download additional
# ML files during Render deployment.
#
# This implementation keeps the application lightweight and
# makes the vector database completely self-contained.


VECTOR_SIZE = 256


def text_to_embedding(text: str):

    vector = [0.0] * VECTOR_SIZE

    words = (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace("-", " ")
        .split()
    )

    if not words:
        return vector

    for word in words:

        digest = hashlib.sha256(
            word.encode("utf-8")
        ).digest()

        index = int.from_bytes(
            digest[:4],
            "big"
        ) % VECTOR_SIZE

        vector[index] += 1.0

    magnitude = sum(
        value * value
        for value in vector
    ) ** 0.5

    if magnitude > 0:

        vector = [
            value / magnitude
            for value in vector
        ]

    return vector


# ============================================================
# 9. CHROMA DATABASE
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

try:

    chroma_client.delete_collection(
        name="schedule"
    )

except Exception:

    pass


collection = chroma_client.get_or_create_collection(
    name="schedule"
)


# ============================================================
# 10. CONVERT EVENT TO TEXT
# ============================================================

def event_to_text(event):

    return (
        f"Title: {event['title']}. "
        f"Type: {event['type']}. "
        f"Date: {event['date']}. "
        f"Time: {event['start_time']} "
        f"to {event['end_time']}. "
        f"Location: {event['location']}. "
        f"Description: {event['description']}."
    )


# ============================================================
# 11. BUILD VECTOR DATABASE
# ============================================================

def build_vector_database():

    schedule = load_schedule()

    if not schedule:
        return

    documents = []
    ids = []
    metadatas = []
    embeddings = []

    for event in schedule:

        text = event_to_text(event)

        documents.append(text)

        ids.append(event["id"])

        metadatas.append({
            "date": event["date"],
            "type": event["type"]
        })

        embeddings.append(
            text_to_embedding(text)
        )

    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
        embeddings=embeddings
    )


# Build database on startup
build_vector_database()


# ============================================================
# 12. REBUILD VECTOR DATABASE
# ============================================================

def rebuild_vector_database():

    try:

        chroma_client.delete_collection(
            name="schedule"
        )

    except Exception:

        pass

    global collection

    collection = chroma_client.get_or_create_collection(
        name="schedule"
    )

    build_vector_database()


# ============================================================
# 13. NORMALIZE DATE
# ============================================================

def normalize_date(value):

    if not value:
        return None

    value = value.strip().lower()

    today = datetime.now().date()

    if value == "today":

        return today.isoformat()

    if value == "tomorrow":

        return (
            today +
            timedelta(days=1)
        ).isoformat()

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    if value in weekdays:

        target = weekdays[value]

        days_ahead = (
            target -
            today.weekday()
        ) % 7

        return (
            today +
            timedelta(days=days_ahead)
        ).isoformat()

    formats = [
        "%Y-%m-%d",
        "%B %d",
        "%b %d",
        "%B %d %Y",
        "%b %d %Y"
    ]

    for fmt in formats:

        try:

            parsed = datetime.strptime(
                value,
                fmt
            )

            if "%Y" not in fmt:

                parsed = parsed.replace(
                    year=today.year
                )

            return parsed.date().isoformat()

        except ValueError:

            continue

    return value


# ============================================================
# 14. GET SCHEDULE TOOL
# ============================================================

@tool
def get_schedule(
    query: str,
    date: Optional[str] = None,
    event_type: Optional[str] = None
) -> str:

    """
    Retrieve relevant schedule information
    using ChromaDB vector retrieval.
    """

    schedule = load_schedule()

    # --------------------------------------------------------
    # DATE FILTER
    # --------------------------------------------------------

    if date:

        normalized_date = normalize_date(
            date
        )

        schedule = [
            event
            for event in schedule
            if event["date"] == normalized_date
        ]

    # --------------------------------------------------------
    # EVENT TYPE FILTER
    # --------------------------------------------------------

    if event_type:

        schedule = [
            event
            for event in schedule
            if event["type"].lower()
            == event_type.lower()
        ]

    if not schedule:

        return "No matching schedule events found."

    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    try:

        query_embedding = text_to_embedding(
            query
        )

        count = collection.count()

        if count > 0:

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(10, count)
            )

            retrieved_ids = (
                results["ids"][0]
                if results.get("ids")
                else []
            )

            event_map = {
                event["id"]: event
                for event in schedule
            }

            vector_results = [
                event_map[event_id]
                for event_id in retrieved_ids
                if event_id in event_map
            ]

            if vector_results:

                schedule = vector_results

    except Exception as error:

        print(
            "Chroma search error:",
            error
        )

    # --------------------------------------------------------
    # FORMAT RESULT
    # --------------------------------------------------------

    result = []

    for event in schedule:

        result.append(
            f"ID: {event['id']}\n"
            f"Title: {event['title']}\n"
            f"Type: {event['type']}\n"
            f"Date: {event['date']}\n"
            f"Time: {event['start_time']} - "
            f"{event['end_time']}\n"
            f"Location: {event['location']}\n"
            f"Description: {event['description']}"
        )

    return "\n\n".join(result)


# ============================================================
# 15. UPDATE SCHEDULE TOOL
# ============================================================

@tool
def update_schedule(
    operation: str,
    event_id: Optional[str] = None,
    title: Optional[str] = None,
    event_type: Optional[str] = None,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None
) -> str:

    """
    Add, update or remove schedule entries.
    """

    schedule = load_schedule()

    operation = operation.lower().strip()

    # ========================================================
    # ADD
    # ========================================================

    if operation == "add":

        if not title:
            return "Title is required."

        if not date:
            return "Date is required."

        if not start_time:
            return "Start time is required."

        normalized_date = normalize_date(date)

        new_event = {

            "id":
                "event-" +
                str(uuid.uuid4())[:8],

            "title":
                title,

            "type":
                event_type or "meeting",

            "date":
                normalized_date,

            "start_time":
                start_time,

            "end_time":
                end_time or "Not specified",

            "location":
                location or "Not specified",

            "description":
                description or ""
        }

        schedule.append(new_event)

        save_schedule(schedule)

        rebuild_vector_database()

        return (
            "Event added successfully.\n\n"
            +
            event_to_text(new_event)
        )

    # ========================================================
    # UPDATE
    # ========================================================

    if operation == "update":

        if not event_id:

            return "event_id is required."

        target = None

        for event in schedule:

            if event["id"] == event_id:

                target = event

                break

        if target is None:

            return "Event not found."

        if title is not None:
            target["title"] = title

        if event_type is not None:
            target["type"] = event_type

        if date is not None:
            target["date"] = normalize_date(date)

        if start_time is not None:
            target["start_time"] = start_time

        if end_time is not None:
            target["end_time"] = end_time

        if location is not None:
            target["location"] = location

        if description is not None:
            target["description"] = description

        save_schedule(schedule)

        rebuild_vector_database()

        return (
            "Event updated successfully.\n\n"
            +
            event_to_text(target)
        )

    # ========================================================
    # REMOVE
    # ========================================================

    if operation == "remove":

        if not event_id:

            return "event_id is required."

        old_length = len(schedule)

        schedule = [
            event
            for event in schedule
            if event["id"] != event_id
        ]

        if len(schedule) == old_length:

            return "Event not found."

        save_schedule(schedule)

        rebuild_vector_database()

        return "Event removed successfully."

    return "Invalid operation."


# ============================================================
# 16. TOOLS
# ============================================================

tools = [
    get_schedule,
    update_schedule
]

tool_map = {
    "get_schedule": get_schedule,
    "update_schedule": update_schedule
}


# ============================================================
# 17. AGENT NODE
# ============================================================

def agent_node(state):

    query = state["user_query"]

    system_prompt = """
You are an Agentic RAG Schedule Assistant.

You manage a user's schedule.

You have exactly two tools:

1. get_schedule
2. update_schedule

RULES:

- For questions about existing events,
  use get_schedule.

- For asking about today's schedule,
  use get_schedule with date="today".

- For asking about tomorrow's schedule,
  use get_schedule with date="tomorrow".

- For adding an event,
  use update_schedule with operation="add".

- For removing an event,
  first use get_schedule if you need to identify
  the event and its event_id.

- For moving or updating an event,
  first use get_schedule to identify the event
  and its event_id.

- Never invent an event_id.

- Always use a tool when schedule information
  is required.

- Be concise.

Return a tool call when a tool is required.
"""

    response = llm.bind_tools(tools).invoke(
        [
            HumanMessage(
                content=
                system_prompt
                +
                "\n\nUSER REQUEST:\n"
                +
                query
            )
        ]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 18. TOOL EXECUTION NODE
# ============================================================

def tool_node(state):

    last_message = state["messages"][-1]

    tool_calls = getattr(
        last_message,
        "tool_calls",
        []
    )

    if not tool_calls:

        return {
            "tool_result":
                "No tool was required."
        }

    outputs = []

    for tool_call in tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call.get(
            "args",
            {}
        )

        selected_tool = tool_map.get(
            tool_name
        )

        if selected_tool is None:

            result = "Unknown tool."

        else:

            try:

                result = selected_tool.invoke(
                    tool_args
                )

            except Exception as error:

                print(
                    "Tool execution error:",
                    error
                )

                result = (
                    "Tool error: "
                    +
                    str(error)
                )

        outputs.append(
            str(result)
        )

    return {
        "tool_result":
            "\n\n".join(outputs)
    }


# ============================================================
# 19. ROUTER
# ============================================================

def route_agent(state):

    last_message = state["messages"][-1]

    tool_calls = getattr(
        last_message,
        "tool_calls",
        []
    )

    if tool_calls:

        return "tools"

    return "final"


# ============================================================
# 20. FINAL ANSWER NODE
# ============================================================

def final_node(state):

    query = state["user_query"]

    tool_result = (
        state.get("tool_result")
        or
        "No schedule information was retrieved."
    )

    final_prompt = f"""
You are a helpful schedule assistant.

USER REQUEST:

{query}

SCHEDULE INFORMATION:

{tool_result}

Give the final answer to the user.

IMPORTANT:

- Answer directly.
- Do not mention tools.
- Do not mention LangGraph.
- Do not mention ChromaDB.
- Do not mention RAG.
- Do not mention internal processing.
- If an event was added, clearly confirm it.
- If an event was updated, clearly confirm it.
- If an event was removed, clearly confirm it.
- If no matching event exists, clearly say so.
- Keep the answer concise.

Return only the final answer.
"""

    response = llm.invoke(
        final_prompt
    )

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if "text" in item:

                    text_parts.append(
                        str(item["text"])
                    )

            else:

                text_parts.append(
                    str(item)
                )

        content = "".join(text_parts)

    return {
        "final_answer":
            str(content)
    }


# ============================================================
# 21. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(
    ScheduleState
)

workflow.add_node(
    "agent",
    agent_node
)

workflow.add_node(
    "tools",
    tool_node
)

workflow.add_node(
    "final",
    final_node
)

workflow.add_edge(
    START,
    "agent"
)

workflow.add_conditional_edges(
    "agent",
    route_agent,
    {
        "tools": "tools",
        "final": "final"
    }
)

workflow.add_edge(
    "tools",
    "final"
)

workflow.add_edge(
    "final",
    END
)

graph = workflow.compile()


# ============================================================
# 22. PLAYGROUND INPUT/OUTPUT
# ============================================================

class PlaygroundInput(BaseModel):

    task: str


class PlaygroundOutput(BaseModel):

    output: str


# ============================================================
# 23. FUNCTION USED BY LANGSERVE
# ============================================================

def run_graph_for_playground(
    input_data: PlaygroundInput
) -> PlaygroundOutput:

    task = input_data.task.strip()

    if not task:

        return PlaygroundOutput(
            output="Please enter a schedule request."
        )

    initial_state = {

        "messages": [
            HumanMessage(
                content=task
            )
        ],

        "user_query":
            task,

        "tool_result":
            None,

        "final_answer":
            None
    }

    try:

        result = graph.invoke(
            initial_state,
            config={
                "recursion_limit": 10
            }
        )

        answer = result.get(
            "final_answer"
        )

        if not answer:

            answer = (
                "No answer was generated."
            )

        return PlaygroundOutput(
            output=str(answer)
        )

    except Exception as error:

        print(
            "Graph execution error:",
            error
        )

        return PlaygroundOutput(
            output=(
                "Application error: "
                +
                str(error)
            )
        )


# ============================================================
# 24. CREATE RUNNABLE
# ============================================================

agent_runnable = RunnableLambda(
    run_graph_for_playground
).with_types(
    input_type=PlaygroundInput,
    output_type=PlaygroundOutput
)


# ============================================================
# 25. FASTAPI
# ============================================================

app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    version="1.0"
)


# ============================================================
# 26. LANGSERVE PLAYGROUND
# ============================================================

add_routes(
    app,
    agent_runnable,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 27. ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message":
            "Agentic RAG Schedule Assistant",
        "playground":
            "/agent/playground/",
        "docs":
            "/docs",
        "health":
            "/health"
    }


# ============================================================
# 28. HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 29. SCHEDULE VIEW
# ============================================================

@app.get("/schedule")
def schedule():

    return {
        "events":
            load_schedule()
    }


# ============================================================
# 30. START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )

