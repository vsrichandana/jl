import os
import json
import uuid
from datetime import datetime, timedelta
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
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


# Current Gemini model can be changed from Render
# Environment Variables.

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
    temperature=0,
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
# 5. CREATE SAMPLE 30-DAY SCHEDULE
# ============================================================

def create_sample_schedule():

    today = datetime.now().date()

    schedule = []


    def add_event(
        title,
        event_type,
        day_offset,
        start_time,
        end_time,
        location,
        description
    ):

        event_date = (
            today
            + timedelta(days=day_offset)
        ).isoformat()


        event = {

            "id": str(uuid.uuid4()),

            "title": title,

            "type": event_type,

            "date": event_date,

            "start_time": start_time,

            "end_time": end_time,

            "location": location,

            "description": description
        }


        schedule.append(event)


    add_event(
        "Team Standup",
        "meeting",
        0,
        "10:00",
        "10:30",
        "Online",
        "Daily project status meeting."
    )


    add_event(
        "DSA Practice",
        "task",
        1,
        "09:00",
        "10:00",
        "Hostel",
        "Practice arrays and sliding window problems."
    )


    add_event(
        "AI Workshop",
        "workshop",
        2,
        "14:00",
        "17:00",
        "College Lab",
        "Hands-on workshop about Agentic AI and RAG."
    )


    add_event(
        "Project Meeting",
        "meeting",
        3,
        "14:00",
        "15:00",
        "Online",
        "Discuss major project progress."
    )


    add_event(
        "Doctor Appointment",
        "appointment",
        5,
        "11:00",
        "12:00",
        "City Clinic",
        "Regular appointment."
    )


    add_event(
        "Java Practice",
        "task",
        7,
        "18:00",
        "19:30",
        "Hostel",
        "Practice Java collections and DSA."
    )


    add_event(
        "Cloud Workshop",
        "workshop",
        10,
        "10:00",
        "13:00",
        "College",
        "Introduction to cloud deployment."
    )


    add_event(
        "Project Review",
        "meeting",
        14,
        "15:00",
        "16:00",
        "Online",
        "Review Agentic RAG implementation."
    )


    add_event(
        "Assignment Submission",
        "task",
        20,
        "12:00",
        "13:00",
        "College Portal",
        "Submit the AI project assignment."
    )


    add_event(
        "Career Workshop",
        "workshop",
        25,
        "16:00",
        "18:00",
        "Seminar Hall",
        "Resume and interview preparation."
    )


    return schedule


# ============================================================
# 6. JSON DATABASE FUNCTIONS
# ============================================================

def initialize_schedule():

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


def load_schedule():

    initialize_schedule()

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


# Initialize JSON database

initialize_schedule()


# ============================================================
# 7. CHROMADB INITIALIZATION
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
# 8. CONVERT SCHEDULE TO TEXT
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
# 9. BUILD VECTOR DATABASE
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
# 10. INITIAL VECTOR DATABASE BUILD
# ============================================================

if collection.count() == 0:

    rebuild_vector_database()


# ============================================================
# 11. DATE NORMALIZATION
# ============================================================

def normalize_date(
    date_text: Optional[str]
):

    if not date_text:

        return None


    text = date_text.strip().lower()


    today = datetime.now().date()


    if text == "today":

        return today.isoformat()


    if text == "tomorrow":

        return (
            today
            + timedelta(days=1)
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

        target_day = weekdays[text]

        days_ahead = (

            target_day
            - today.weekday()
        ) % 7


        return (

            today
            + timedelta(
                days=days_ahead
            )

        ).isoformat()


    # YYYY-MM-DD

    try:

        parsed = datetime.strptime(
            text,
            "%Y-%m-%d"
        )

        return parsed.date().isoformat()

    except ValueError:

        pass


    # August 15 / Aug 15

    for fmt in [
        "%B %d",
        "%b %d"
    ]:

        try:

            parsed = datetime.strptime(
                text,
                fmt
            )

            return parsed.replace(
                year=today.year
            ).date().isoformat()

        except ValueError:

            pass


    return text


# ============================================================
# 12. TIME NORMALIZATION
# ============================================================

def normalize_time(
    time_text: Optional[str]
):

    if not time_text:

        return None


    text = (
        time_text
        .strip()
        .upper()
    )


    formats = [

        "%I %p",

        "%I:%M %p",

        "%H:%M"
    ]


    for fmt in formats:

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
# 13. SCHEDULE SEARCH
# ============================================================

def search_schedule(

    query: str,

    date: Optional[str] = None,

    event_type: Optional[str] = None

):

    schedule = load_schedule()


    normalized_date = normalize_date(
        date
    )


    filtered = schedule


    if normalized_date:

        filtered = [

            event

            for event in filtered

            if event["date"]
            == normalized_date
        ]


    if event_type:

        filtered = [

            event

            for event in filtered

            if event["type"].lower()
            == event_type.lower()
        ]


    # --------------------------------------------------------
    # ChromaDB semantic retrieval
    # --------------------------------------------------------

    try:

        result = collection.query(

            query_texts=[query],

            n_results=min(
                10,
                max(
                    collection.count(),
                    1
                )
            )
        )


        retrieved_ids = (

            result["ids"][0]
            if result.get("ids")
            else []
        )


        id_map = {

            event["id"]: event

            for event in filtered
        }


        semantic_results = [

            id_map[event_id]

            for event_id in retrieved_ids

            if event_id in id_map
        ]


        if semantic_results:

            filtered = semantic_results


    except Exception:

        pass


    return filtered


# ============================================================
# 14. FORMAT SCHEDULE
# ============================================================

def format_schedule(events):

    if not events:

        return (
            "No schedule events were found."
        )


    result = []


    for event in events:

        result.append(

            f"ID: {event['id']}\n"

            f"Title: {event['title']}\n"

            f"Type: {event['type']}\n"

            f"Date: {event['date']}\n"

            f"Time: "
            f"{event['start_time']} - "
            f"{event['end_time']}\n"

            f"Location: "
            f"{event['location']}\n"

            f"Description: "
            f"{event['description']}"
        )


    return "\n\n".join(result)


# ============================================================
# 15. TOOL 1 - GET SCHEDULE
# ============================================================

@tool
def get_schedule(

    query: str,

    date: Optional[str] = None,

    event_type: Optional[str] = None

) -> str:

    """
    Retrieve relevant schedule information.

    Use this tool when the user asks about:

    - today's schedule
    - tomorrow's schedule
    - a particular date
    - meetings
    - workshops
    - tasks
    - appointments
    - free time
    - availability
    - existing events
    """


    events = search_schedule(

        query=query,

        date=date,

        event_type=event_type
    )


    return format_schedule(
        events
    )


# ============================================================
# 16. TOOL 2 - UPDATE SCHEDULE
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
    Add, update, or remove schedule entries.

    operation must be:

    add
    update
    remove
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


        normalized_date = normalize_date(
            date
        )


        normalized_start = normalize_time(
            start_time
        )


        if end_time:

            normalized_end = normalize_time(
                end_time
            )

        else:

            start_dt = datetime.strptime(

                normalized_start,

                "%H:%M"
            )


            normalized_end = (

                start_dt
                + timedelta(
                    hours=1
                )

            ).strftime(
                "%H:%M"
            )


        new_event = {

            "id": str(
                uuid.uuid4()
            ),

            "title": title,

            "type":
                event_type
                or "meeting",

            "date":
                normalized_date,

            "start_time":
                normalized_start,

            "end_time":
                normalized_end,

            "location":
                location
                or "Not specified",

            "description":
                description
                or ""
        }


        schedule.append(
            new_event
        )


        save_schedule(
            schedule
        )


        rebuild_vector_database()


        return (

            "Schedule event added "
            "successfully.\n\n"

            + event_to_text(
                new_event
            )
        )


    # ========================================================
    # UPDATE
    # ========================================================

    if operation.lower() == "update":

        if not event_id:

            return (
                "event_id is required "
                "for update."
            )


        target_event = None


        for event in schedule:

            if event["id"] == event_id:

                target_event = event

                break


        if target_event is None:

            return (
                "Event not found."
            )


        if title is not None:

            target_event["title"] = title


        if event_type is not None:

            target_event["type"] = event_type


        if date is not None:

            target_event["date"] = (
                normalize_date(date)
            )


        if start_time is not None:

            target_event["start_time"] = (
                normalize_time(
                    start_time
                )
            )


        if end_time is not None:

            target_event["end_time"] = (
                normalize_time(
                    end_time
                )
            )


        if location is not None:

            target_event["location"] = location


        if description is not None:

            target_event["description"] = (
                description
            )


        save_schedule(
            schedule
        )


        rebuild_vector_database()


        return (

            "Schedule event updated "
            "successfully.\n\n"

            + event_to_text(
                target_event
            )
        )


    # ========================================================
    # REMOVE
    # ========================================================

    if operation.lower() == "remove":

        if not event_id:

            return (
                "event_id is required "
                "for removal."
            )


        original_length = len(
            schedule
        )


        schedule = [

            event

            for event in schedule

            if event["id"] != event_id
        ]


        if len(schedule) == original_length:

            return (
                "Event not found."
            )


        save_schedule(
            schedule
        )


        rebuild_vector_database()


        return (
            "Schedule event removed "
            "successfully."
        )


    return (
        "Invalid operation. "
        "Use add, update, or remove."
    )


# ============================================================
# 17. TOOL LIST
# ============================================================

schedule_tools = [

    get_schedule,

    update_schedule
]


# ============================================================
# 18. GRAPH NODE - INPUT
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
# 19. AGENT NODE
# ============================================================

def schedule_agent_node(
    state: ScheduleState
):

    print("\n" + "=" * 50)

    print(
        "             SCHEDULE AGENT"
    )

    print("=" * 50)


    user_query = state[
        "user_query"
    ]


    system_prompt = """

You are an Agentic RAG Schedule Assistant.

You manage a user's schedule
for the next 30 days.

You have exactly two tools:

1. get_schedule

2. update_schedule


IMPORTANT:

You must decide which tool is
appropriate based on the user's request.


For questions about existing
schedule information:

Use get_schedule.


For adding, changing, or removing
schedule entries:

Use update_schedule.


If the user asks:

"What do I have scheduled tomorrow?"

Use get_schedule.


If the user asks:

"Am I free Friday afternoon?"

Use get_schedule.


If the user asks:

"Add a meeting on August 15
at 3 PM."

Use update_schedule.


If the user asks:

"Move my meeting from 2 PM
to 4 PM."

First use get_schedule to identify
the meeting.

Then use update_schedule to change
the event.


Never invent schedule information.


For availability questions:

If get_schedule returns no event
during the requested period,
the user is free during that period.


If an event exists during that
period, tell the user that the
time is occupied.


When adding an event and no end
time is provided, assume one hour.


Always give a concise and clear
final answer.
"""


    messages = [

        HumanMessage(
            content=
                system_prompt
                + "\n\nUSER REQUEST:\n"
                + user_query
        )
    ]


    response = llm.bind_tools(
        schedule_tools
    ).invoke(messages)


    return {

        "messages": [
            response
        ],

        "next_step":
            "tool_or_finish"
    }


# ============================================================
# 20. TOOL EXECUTION NODE
# ============================================================

def execute_schedule_tool(
    state: ScheduleState
):

    print("\n" + "=" * 50)

    print(
        "              TOOL EXECUTION"
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
                "completed"
        }


    results = []


    for tool_call in tool_calls:

        tool_name = (
            tool_call["name"]
        )


        tool_args = (
            tool_call["args"]
        )


        print(
            f"[Agent] Calling tool: "
            f"{tool_name}"
        )


        if tool_name == "get_schedule":

            result = (
                get_schedule.invoke(
                    tool_args
                )
            )


            results.append(
                result
            )


        elif tool_name == "update_schedule":

            result = (
                update_schedule.invoke(
                    tool_args
                )
            )


            results.append(
                result
            )


    combined_result = (
        "\n\n".join(results)
    )


    return {

        "retrieved_schedule":
            combined_result,

        "operation_result":
            combined_result,

        "next_step":
            "final"
    }


# ============================================================
# 21. FINAL RESPONSE NODE
# ============================================================

def final_response_node(
    state: ScheduleState
):

    print("\n" + "=" * 50)

    print(
        "             FINAL RESPONSE"
    )

    print("=" * 50)


    user_query = state[
        "user_query"
    ]


    tool_result = (

        state.get(
            "operation_result"
        )

        or

        state.get(
            "retrieved_schedule"
        )

        or

        "No tool result."
    )


    final_prompt = f"""

You are a helpful schedule assistant.

User request:

{user_query}


Retrieved schedule information
or operation result:

{tool_result}


Answer the user clearly.

Do not invent events.

If no matching event was found,
say that the user is free during
the requested period.

If an event was added, updated,
or removed, confirm the change.

Keep the response concise.
"""


    response = llm.invoke(
        final_prompt
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


        final_answer = "\n".join(
            parts
        )


    else:

        final_answer = str(
            content
        )


    return {

        "final_answer":
            final_answer,

        "next_step":
            "completed"
    }


# ============================================================
# 22. GRAPH ROUTING
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
# 23. GRAPH CONSTRUCTION
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
    schedule_agent_node
)


workflow.add_node(
    "tools",
    execute_schedule_tool
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


workflow.add_edge(
    "tools",
    "final"
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
# 24. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=
        "Agentic RAG Schedule Assistant",

    description=
        "30-Day Schedule Assistant "
        "using Gemini, LangGraph, "
        "ChromaDB and RAG.",

    version="1.0.0"
)


# ============================================================
# 25. FUNCTION FOR LANGSERVE PLAYGROUND
# ============================================================

def run_graph_for_playground(
    task: str
) -> str:

    """
    Convert simple string input
    from LangServe playground into
    ScheduleState.
    """


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
                50
        }
    )


    return result.get(
        "final_answer",
        "No response generated."
    )


# ============================================================
# 26. CONVERT FUNCTION INTO RUNNABLE
# ============================================================

agent_runnable = RunnableLambda(
    run_graph_for_playground
)


# ============================================================
# 27. LANGSERVE PLAYGROUND
# ============================================================

add_routes(

    app,

    agent_runnable,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 28. API REQUEST MODEL
# ============================================================

class ScheduleRequest(
    BaseModel
):

    task: str


# ============================================================
# 29. HEALTH CHECK
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

        "agent":
            "/agent/playground/"
    }


# ============================================================
# 30. TEST AGENT
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
# 31. VIEW SCHEDULE
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
# 32. CUSTOM SCHEDULE ENDPOINT
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
                50
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
# 33. START SERVER
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
