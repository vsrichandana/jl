import os
import json
import uuid

from datetime import datetime, timedelta
from typing import TypedDict, List, Optional, Annotated

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage
)

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.graph.message import add_messages

from langserve import add_routes

import chromadb


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# ============================================================
# 2. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
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

    if not os.path.exists(
        SCHEDULE_FILE
    ):

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


    with open(
        SCHEDULE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


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
# 8. CHROMA DATABASE
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


collection = (
    chroma_client
    .get_or_create_collection(
        name="schedule"
    )
)


# ============================================================
# 9. CONVERT EVENT TO TEXT
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
# 10. ADD SCHEDULE TO CHROMA
# ============================================================

def build_vector_database():

    schedule = load_schedule()


    if not schedule:

        return


    documents = []

    ids = []

    metadatas = []


    for event in schedule:

        documents.append(
            event_to_text(event)
        )

        ids.append(
            event["id"]
        )

        metadatas.append({

            "date":
                event["date"],

            "type":
                event["type"]
        })


    collection.add(

        documents=documents,

        ids=ids,

        metadatas=metadatas
    )


# Build Chroma when server starts

build_vector_database()


# ============================================================
# 11. NORMALIZE DATE
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
# 12. GET SCHEDULE TOOL
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

            if event["date"] ==
            normalized_date
        ]


    # --------------------------------------------------------
    # EVENT TYPE FILTER
    # --------------------------------------------------------

    if event_type:

        schedule = [

            event

            for event in schedule

            if event["type"].lower()
            ==
            event_type.lower()
        ]


    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    try:

        if collection.count() > 0:

            results = collection.query(

                query_texts=[query],

                n_results=min(
                    10,
                    collection.count()
                )
            )


            retrieved_ids = (

                results["ids"][0]

                if results.get("ids")
                else []
            )


            event_map = {

                event["id"]:
                    event

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


    if not schedule:

        return (
            "No matching schedule events found."
        )


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
            f"Description: "
            f"{event['description']}"
        )


    return "\n\n".join(result)


# ============================================================
# 13. UPDATE SCHEDULE TOOL
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


    # ========================================================
    # ADD
    # ========================================================

    if operation.lower() == "add":

        if not title:

            return (
                "Title is required."
            )


        if not date:

            return (
                "Date is required."
            )


        if not start_time:

            return (
                "Start time is required."
            )


        new_event = {

            "id":
                "event-" +
                str(uuid.uuid4())[:8],

            "title":
                title,

            "type":
                event_type or "meeting",

            "date":
                normalize_date(date),

            "start_time":
                start_time,

            "end_time":
                end_time or "Not specified",

            "location":
                location or "Not specified",

            "description":
                description or ""
        }


        schedule.append(
            new_event
        )


        save_schedule(
            schedule
        )


        return (

            "Event added successfully.\n\n"

            +
            event_to_text(new_event)
        )


    # ========================================================
    # UPDATE
    # ========================================================

    if operation.lower() == "update":

        if not event_id:

            return (
                "event_id is required."
            )


        target = None


        for event in schedule:

            if event["id"] == event_id:

                target = event

                break


        if target is None:

            return (
                "Event not found."
            )


        if title is not None:

            target["title"] = title


        if event_type is not None:

            target["type"] = event_type


        if date is not None:

            target["date"] = normalize_date(
                date
            )


        if start_time is not None:

            target["start_time"] = start_time


        if end_time is not None:

            target["end_time"] = end_time


        if location is not None:

            target["location"] = location


        if description is not None:

            target["description"] = description


        save_schedule(
            schedule
        )


        return (

            "Event updated successfully.\n\n"

            +
            event_to_text(target)
        )


    # ========================================================
    # REMOVE
    # ========================================================

    if operation.lower() == "remove":

        if not event_id:

            return (
                "event_id is required."
            )


        old_length = len(
            schedule
        )


        schedule = [

            event

            for event in schedule

            if event["id"] != event_id
        ]


        if len(schedule) == old_length:

            return (
                "Event not found."
            )


        save_schedule(
            schedule
        )


        return (
            "Event removed successfully."
        )


    return (
        "Invalid operation."
    )


# ============================================================
# 14. TOOLS
# ============================================================

tools = [

    get_schedule,

    update_schedule
]


tool_map = {

    "get_schedule":
        get_schedule,

    "update_schedule":
        update_schedule
}


# ============================================================
# 15. AGENT NODE
# ============================================================

def agent_node(state):

    query = state["user_query"]


    system_prompt = """

You are an Agentic RAG Schedule Assistant.

You manage a user's schedule.

You have exactly two tools:

1. get_schedule
2. update_schedule

Rules:

- For questions about existing events,
  use get_schedule.

- For adding an event,
  use update_schedule.

- For removing an event,
  first use get_schedule if you need
  to identify the event.

- For moving or updating an event,
  first use get_schedule to identify
  the event and its event_id.

- Never invent an event_id.

- If schedule information is needed,
  use the appropriate tool.

After the tool result is provided,
the application will generate the final
answer.

Be concise.
"""


    response = (

        llm

        .bind_tools(tools)

        .invoke([

            HumanMessage(
                content=
                system_prompt
                +
                "\n\nUSER REQUEST:\n"
                +
                query
            )

        ])
    )


    return {

        "messages":
            [response]
    }


# ============================================================
# 16. TOOL EXECUTION NODE
# ============================================================

def tool_node(state):

    last_message = (
        state["messages"][-1]
    )


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

        tool_name = (
            tool_call["name"]
        )


        tool_args = (
            tool_call["args"]
        )


        selected_tool = (
            tool_map.get(
                tool_name
            )
        )


        if selected_tool is None:

            result = (
                "Unknown tool."
            )

        else:

            try:

                result = (
                    selected_tool.invoke(
                        tool_args
                    )
                )

            except Exception as error:

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
# 17. ROUTER
# ============================================================

def route_agent(state):

    last_message = (
        state["messages"][-1]
    )


    tool_calls = getattr(
        last_message,
        "tool_calls",
        []
    )


    if tool_calls:

        return "tools"


    return "final"


# ============================================================
# 18. FINAL ANSWER NODE
# ============================================================

def final_node(state):

    query = (
        state["user_query"]
    )


    tool_result = (
        state.get(
            "tool_result"
        )
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

Important:

- Answer directly.
- Do not mention tools.
- Do not mention LangGraph.
- Do not mention ChromaDB.
- Do not mention RAG.
- Do not mention internal processing.
- If an event was added, updated or removed,
  clearly confirm it.
- If no matching event exists,
  clearly say so.

Return only the final answer.
"""


    response = llm.invoke(
        final_prompt
    )


    return {

        "final_answer":
            str(response.content)
    }


# ============================================================
# 19. BUILD LANGGRAPH
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

        "tools":
            "tools",

        "final":
            "final"
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
# 20. FUNCTION USED BY LANGSERVE PLAYGROUND
# ============================================================

def run_graph_for_playground(
    task: str
) -> str:

    """

    This function receives the text entered
    in the LangServe Playground.

    It runs the LangGraph and RETURNS A STRING.

    Returning a completed string is important
    because the LangServe Playground displays
    this returned value.
    """

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


    result = graph.invoke(

        initial_state,

        config={

            "recursion_limit":
                5
        }
    )


    answer = result.get(
        "final_answer"
    )


    if answer is None:

        return (
            "No answer was generated."
        )


    return str(answer)


# ============================================================
# 21. CREATE RUNNABLE
# ============================================================

agent_runnable = RunnableLambda(
    run_graph_for_playground
)


# ============================================================
# 22. FASTAPI
# ============================================================

app = FastAPI(

    title=
        "Agentic RAG Schedule Assistant",

    version=
        "1.0"
)


# ============================================================
# 23. LANGSERVE PLAYGROUND
# ============================================================

add_routes(

    app,

    agent_runnable,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 24. ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "running",

        "message":
            "Agentic RAG Schedule Assistant",

        "playground":
            "/agent/playground/",

        "docs":
            "/docs"
    }


# ============================================================
# 25. HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy"
    }


# ============================================================
# 26. SCHEDULE VIEW
# ============================================================

@app.get("/schedule")
def schedule():

    return {

        "events":
            load_schedule()
    }


# ============================================================
# 27. START SERVER
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
