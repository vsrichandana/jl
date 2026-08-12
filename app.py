import os
import json
import uuid
from datetime import datetime, timedelta
from typing import TypedDict, List, Optional

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

from langgraph.graph import StateGraph, START, END

from langserve import add_routes

import chromadb


# ============================================================
# 1. CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )


GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite-preview"
)


# ============================================================
# 2. LLM INITIALIZATION
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. STATE DEFINITION
# ============================================================

class ScheduleState(TypedDict):

    messages: List[BaseMessage]

    next_step: Optional[str]

    user_query: Optional[str]

    retrieved_schedule: Optional[str]

    operation_result: Optional[str]

    final_answer: Optional[str]


# ============================================================
# 4. FILE PATHS
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
# 5. JSON DATABASE
# ============================================================

def create_initial_schedule():

    today = datetime.now().date()

    schedule = []


    def add_event(
        event_id,
        title,
        event_type,
        day_offset,
        start_time,
        end_time,
        location,
        description
    ):

        event_date = (
            today +
            timedelta(days=day_offset)
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
        "Practice arrays and sliding window problems."
    )


    add_event(
        "event-003",
        "AI Workshop",
        "workshop",
        2,
        "14:00",
        "17:00",
        "College Lab",
        "Hands-on workshop about Agentic AI and RAG."
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
        "Practice Java collections and DSA."
    )


    add_event(
        "event-007",
        "Cloud Workshop",
        "workshop",
        10,
        "10:00",
        "13:00",
        "College",
        "Introduction to cloud deployment."
    )


    add_event(
        "event-008",
        "Project Review",
        "meeting",
        14,
        "15:00",
        "16:00",
        "Online",
        "Review Agentic RAG implementation."
    )


    add_event(
        "event-009",
        "Assignment Submission",
        "task",
        20,
        "12:00",
        "13:00",
        "College Portal",
        "Submit the AI project assignment."
    )


    add_event(
        "event-010",
        "Career Workshop",
        "workshop",
        25,
        "16:00",
        "18:00",
        "Seminar Hall",
        "Resume and interview preparation."
    )


    return schedule


def load_schedule():

    if not os.path.exists(
        SCHEDULE_FILE
    ):

        schedule = create_initial_schedule()

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
# 6. CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)


collection = (
    chroma_client.get_or_create_collection(
        name="schedule"
    )
)


# ============================================================
# 7. EVENT TO TEXT
# ============================================================

def event_to_text(event):

    return (

        f"Event ID: {event['id']}. "

        f"Title: {event['title']}. "

        f"Type: {event['type']}. "

        f"Date: {event['date']}. "

        f"Time: "
        f"{event['start_time']} "
        f"to "
        f"{event['end_time']}. "

        f"Location: "
        f"{event['location']}. "

        f"Description: "
        f"{event['description']}."
    )


# ============================================================
# 8. REBUILD VECTOR DATABASE
# ============================================================

def rebuild_vector_database():

    global collection


    try:

        chroma_client.delete_collection(
            name="schedule"
        )

    except Exception:

        pass


    collection = (
        chroma_client.get_or_create_collection(
            name="schedule"
        )
    )


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
                event["type"],

            "start_time":
                event["start_time"],

            "end_time":
                event["end_time"]
        })


    collection.add(

        documents=documents,

        ids=ids,

        metadatas=metadatas
    )


# ============================================================
# 9. INITIALIZE VECTOR DATABASE
# ============================================================

load_schedule()


if collection.count() == 0:

    rebuild_vector_database()


# ============================================================
# 10. DATE NORMALIZATION
# ============================================================

def normalize_date(date_text):

    if not date_text:

        return None


    text = (
        date_text
        .strip()
        .lower()
    )


    today = datetime.now().date()


    if text == "today":

        return today.isoformat()


    if text == "tomorrow":

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


    if text in weekdays:

        target = weekdays[text]

        days_ahead = (
            target -
            today.weekday()
        ) % 7


        return (
            today +
            timedelta(days=days_ahead)
        ).isoformat()


    for fmt in [
        "%Y-%m-%d",
        "%B %d",
        "%b %d"
    ]:

        try:

            parsed = datetime.strptime(
                text,
                fmt
            )


            if fmt != "%Y-%m-%d":

                parsed = parsed.replace(
                    year=today.year
                )


            return parsed.date().isoformat()

        except ValueError:

            pass


    return text


# ============================================================
# 11. TIME NORMALIZATION
# ============================================================

def normalize_time(time_text):

    if not time_text:

        return None


    text = (
        time_text
        .strip()
        .upper()
    )


    for fmt in [
        "%I %p",
        "%I:%M %p",
        "%H:%M"
    ]:

        try:

            parsed = datetime.strptime(
                text,
                fmt
            )


            return parsed.strftime(
                "%H:%M"
            )

        except ValueError:

            pass


    return text


# ============================================================
# 12. RETRIEVAL
# ============================================================

def retrieve_schedule(
    query,
    date=None,
    event_type=None
):

    schedule = load_schedule()


    normalized_date = (
        normalize_date(date)
    )


    # --------------------------------------------------------
    # First filter by exact metadata
    # --------------------------------------------------------

    candidates = schedule


    if normalized_date:

        candidates = [

            event

            for event in candidates

            if event["date"]
            == normalized_date
        ]


    if event_type:

        candidates = [

            event

            for event in candidates

            if event["type"].lower()
            == event_type.lower()
        ]


    # --------------------------------------------------------
    # Vector retrieval
    # --------------------------------------------------------

    try:

        if collection.count() > 0:

            result = collection.query(

                query_texts=[query],

                n_results=min(
                    10,
                    collection.count()
                )
            )


            ids = (
                result["ids"][0]
                if result.get("ids")
                else []
            )


            candidate_map = {

                event["id"]: event

                for event in candidates
            }


            semantic_results = [

                candidate_map[event_id]

                for event_id in ids

                if event_id in candidate_map
            ]


            if semantic_results:

                candidates = semantic_results


    except Exception as error:

        print(
            "Vector retrieval error:",
            error
        )


    return candidates


# ============================================================
# 13. FORMAT EVENTS
# ============================================================

def format_events(events):

    if not events:

        return (
            "No matching schedule events found."
        )


    output = []


    for event in events:

        output.append(

            f"ID: {event['id']}\n"
            f"Title: {event['title']}\n"
            f"Type: {event['type']}\n"
            f"Date: {event['date']}\n"
            f"Time: "
            f"{event['start_time']} - "
            f"{event['end_time']}\n"
            f"Location: {event['location']}\n"
            f"Description: "
            f"{event['description']}"
        )


    return "\n\n".join(output)


# ============================================================
# 14. TOOL 1 - GET SCHEDULE
# ============================================================

@tool
def get_schedule(
    query: str,
    date: Optional[str] = None,
    event_type: Optional[str] = None
) -> str:

    """
    Retrieve relevant schedule information.

    Use this for questions about existing
    meetings, workshops, tasks,
    appointments, dates, times,
    and availability.
    """


    events = retrieve_schedule(

        query=query,

        date=date,

        event_type=event_type
    )


    return format_events(events)


# ============================================================
# 15. TOOL 2 - UPDATE SCHEDULE
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

    operation must be add, update or remove.
    """


    schedule = load_schedule()


    # ========================================================
    # ADD
    # ========================================================

    if operation.lower() == "add":

        if not title:

            return "Title is required."


        if not date:

            return "Date is required."


        if not start_time:

            return "Start time is required."


        normalized_date = (
            normalize_date(date)
        )


        normalized_start = (
            normalize_time(start_time)
        )


        if end_time:

            normalized_end = (
                normalize_time(end_time)
            )

        else:

            start_dt = datetime.strptime(
                normalized_start,
                "%H:%M"
            )


            normalized_end = (
                start_dt +
                timedelta(hours=1)
            ).strftime("%H:%M")


        new_event = {

            "id":
                str(uuid.uuid4()),

            "title":
                title,

            "type":
                event_type or "meeting",

            "date":
                normalized_date,

            "start_time":
                normalized_start,

            "end_time":
                normalized_end,

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


        rebuild_vector_database()


        return (
            "Event added successfully.\n\n"
            + event_to_text(new_event)
        )


    # ========================================================
    # UPDATE
    # ========================================================

    if operation.lower() == "update":

        if not event_id:

            return (
                "event_id is required for update."
            )


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

            target["date"] = (
                normalize_date(date)
            )


        if start_time is not None:

            target["start_time"] = (
                normalize_time(start_time)
            )


        if end_time is not None:

            target["end_time"] = (
                normalize_time(end_time)
            )


        if location is not None:

            target["location"] = location


        if description is not None:

            target["description"] = description


        save_schedule(
            schedule
        )


        rebuild_vector_database()


        return (
            "Event updated successfully.\n\n"
            + event_to_text(target)
        )


    # ========================================================
    # REMOVE
    # ========================================================

    if operation.lower() == "remove":

        if not event_id:

            return (
                "event_id is required for removal."
            )


        old_length = len(schedule)


        schedule = [

            event

            for event in schedule

            if event["id"] != event_id
        ]


        if len(schedule) == old_length:

            return "Event not found."


        save_schedule(
            schedule
        )


        rebuild_vector_database()


        return (
            "Event removed successfully."
        )


    return (
        "Invalid operation. "
        "Use add, update or remove."
    )


# ============================================================
# 16. TOOLS
# ============================================================

schedule_tools = [

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
# 17. INPUT NODE
# ============================================================

def schedule_input_node(
    state: ScheduleState
):

    user_query = (
        state["messages"][-1].content
    )


    return {

        "user_query":
            user_query,

        "next_step":
            "agent"
    }


# ============================================================
# 18. AGENT NODE
# ============================================================

def agent_node(
    state: ScheduleState
):

    print("\n" + "=" * 50)

    print(
        "              AGENT"
    )

    print("=" * 50)


    system_prompt = """

You are an Agentic RAG Schedule Assistant.

You manage the user's schedule.

You have two tools:

1. get_schedule
2. update_schedule

Use get_schedule when the user asks
about existing schedule information.

Use update_schedule when the user wants
to add, change or remove an event.

For a request such as:

"Move my meeting from 2 PM to 4 PM."

You should first call get_schedule
to identify the meeting.

After receiving the result, call
update_schedule with the correct
event_id.

For availability questions, retrieve
the relevant schedule first.

Never invent events or event IDs.

If the user says "tomorrow", "today",
or a weekday, interpret it relative
to the current date.

When adding an event without an end
time, use a one-hour duration.

Give a concise final response.
"""


    messages = [

        HumanMessage(
            content=system_prompt
        )
    ]


    # --------------------------------------------------------
    # Add previous conversation/tool results
    # --------------------------------------------------------

    for message in state["messages"]:

        messages.append(message)


    response = (
        llm
        .bind_tools(schedule_tools)
        .invoke(messages)
    )


    print(
        "[Agent] Response generated."
    )


    return {

        "messages":
            [response],

        "next_step":
            "tools"
    }


# ============================================================
# 19. TOOL NODE
# ============================================================

def tool_node(
    state: ScheduleState
):

    print("\n" + "=" * 50)

    print(
        "             TOOL EXECUTION"
    )

    print("=" * 50)


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

            "next_step":
                "final"
        }


    new_messages = []


    for tool_call in tool_calls:

        tool_name = (
            tool_call["name"]
        )


        tool_args = (
            tool_call["args"]
        )


        print(
            f"[Tool] {tool_name}"
        )


        selected_tool = (
            tool_map.get(tool_name)
        )


        if selected_tool is None:

            result = (
                f"Unknown tool: "
                f"{tool_name}"
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
                    "Tool execution error: "
                    + str(error)
                )


        new_messages.append(

            ToolMessage(

                content=str(result),

                tool_call_id=
                    tool_call["id"]
            )
        )


        if tool_name == "get_schedule":

            state_result = str(result)

        else:

            state_result = str(result)


        print(
            "[Tool] Completed."
        )


    return {

        "messages":
            new_messages,

        "retrieved_schedule":
            state_result,

        "operation_result":
            state_result,

        "next_step":
            "agent"
    }


# ============================================================
# 20. ROUTING
# ============================================================

def route_after_agent(
    state: ScheduleState
):

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
# 21. FINAL RESPONSE
# ============================================================

def final_response_node(
    state: ScheduleState
):

    print("\n" + "=" * 50)

    print(
        "             FINAL RESPONSE"
    )

    print("=" * 50)


    messages = state["messages"]


    final_prompt = """

You are the final response generator
for a schedule assistant.

Answer the user's original request
using the tool results in the
conversation.

Do not invent information.

If an event was added, updated,
or removed, clearly confirm it.

If a schedule query returned no
matching event, explain that no
matching event was found.

For availability questions,
say whether the requested period
appears free or occupied based on
the retrieved schedule.

Keep the response concise.
"""


    final_messages = [

        HumanMessage(
            content=final_prompt
        )
    ]


    for message in messages:

        final_messages.append(
            message
        )


    response = llm.invoke(
        final_messages
    )


    content = response.content


    if isinstance(
        content,
        list
    ):

        parts = []


        for item in content:

            if isinstance(
                item,
                dict
            ):

                parts.append(
                    item.get(
                        "text",
                        ""
                    )
                )

            else:

                parts.append(
                    str(item)
                )


        answer = "\n".join(parts)

    else:

        answer = str(content)


    return {

        "final_answer":
            answer,

        "next_step":
            "completed"
    }


# ============================================================
# 22. GRAPH
# ============================================================

workflow = StateGraph(
    ScheduleState
)


workflow.add_node(
    "schedule_input",
    schedule_input_node
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
    final_response_node
)


workflow.add_edge(
    START,
    "schedule_input"
)


workflow.add_edge(
    "schedule_input",
    "agent"
)


workflow.add_conditional_edges(

    "agent",

    route_after_agent,

    {

        "tools":
            "tools",

        "final":
            "final"
    }
)


# ------------------------------------------------------------
# IMPORTANT:
# After a tool call, go back to the agent.
# This allows multiple tool calls.
# ------------------------------------------------------------

workflow.add_edge(
    "tools",
    "agent"
)


workflow.add_edge(
    "final",
    END
)


rt_app = workflow.compile()


print(
    "LangGraph Schedule Agent "
    "compiled successfully."
)


# ============================================================
# 23. FASTAPI
# ============================================================

app = FastAPI(

    title=
        "Agentic RAG Schedule Assistant",

    description=
        "30-Day Schedule Assistant "
        "using Gemini, LangGraph "
        "and ChromaDB.",

    version="1.0.0"
)


# ============================================================
# 24. LANGSERVE FUNCTION
# ============================================================

def run_graph_for_playground(
    task: str
) -> str:

    initial_state: ScheduleState = {

        "messages": [

            HumanMessage(
                content=task
            )
        ],

        "next_step": None,

        "user_query": None,

        "retrieved_schedule": None,

        "operation_result": None,

        "final_answer": None
    }


    result = rt_app.invoke(

        initial_state,

        config={

            "recursion_limit":
                20
        }
    )


    return result.get(

        "final_answer",

        "No response generated."
    )


# ============================================================
# 25. RUNNABLE
# ============================================================

agent_runnable = RunnableLambda(
    run_graph_for_playground
)


# ============================================================
# 26. LANGSERVE
# ============================================================

add_routes(

    app,

    agent_runnable,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 27. REQUEST MODEL
# ============================================================

class ScheduleRequest(
    BaseModel
):

    task: str


# ============================================================
# 28. ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "running",

        "message":
            "Agentic RAG Schedule "
            "Assistant is running",

        "docs":
            "/docs",

        "playground":
            "/agent/playground/"
    }


# ============================================================
# 29. TEST AGENT
# ============================================================

@app.get(
    "/test-agent"
)
def test_agent():

    result = (
        run_graph_for_playground(

            "What do I have scheduled tomorrow?"
        )
    )


    return {

        "result":
            result
    }


# ============================================================
# 30. VIEW SCHEDULE
# ============================================================

@app.get(
    "/schedule"
)
def view_schedule():

    return {

        "events":
            load_schedule()
    }


# ============================================================
# 31. CUSTOM ENDPOINT
# ============================================================

@app.post(
    "/run"
)
def run_agent(
    request: ScheduleRequest
):

    initial_state: ScheduleState = {

        "messages": [

            HumanMessage(
                content=request.task
            )
        ],

        "next_step": None,

        "user_query": None,

        "retrieved_schedule": None,

        "operation_result": None,

        "final_answer": None
    }


    result = rt_app.invoke(

        initial_state,

        config={

            "recursion_limit":
                20
        }
    )


    return {

        "task":
            request.task,

        "answer":
            result.get(
                "final_answer"
            ),

        "retrieved_schedule":
            result.get(
                "retrieved_schedule"
            ),

        "operation_result":
            result.get(
                "operation_result"
            )
    }


# ============================================================
# 32. START SERVER
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
