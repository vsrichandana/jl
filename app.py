import os
import json
import uuid
from datetime import datetime, timedelta, date as date_type
from typing import TypedDict, List, Optional, Any

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langserve import add_routes
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Add it in Render -> Environment Variables."
    )

# Set this in Render if your Gemini account uses a different model.
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite-preview",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_FILE = os.path.join(BASE_DIR, "schedule.json")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# 2. SCHEDULE DATA
# ============================================================

def create_sample_schedule():
    today = datetime.now().date()

    sample = [
        ("Team Standup", "meeting", 0, "10:00", "10:30",
         "Online", "Daily project status meeting."),
        ("DSA Practice", "task", 1, "09:00", "10:00",
         "Hostel", "Practice arrays and sliding window problems."),
        ("AI Workshop", "workshop", 2, "14:00", "17:00",
         "College Lab", "Hands-on workshop about Agentic AI and RAG."),
        ("Project Meeting", "meeting", 3, "14:00", "15:00",
         "Online", "Discuss major project progress."),
        ("Doctor Appointment", "appointment", 5, "11:00", "12:00",
         "City Clinic", "Regular appointment."),
        ("Java Practice", "task", 7, "18:00", "19:30",
         "Hostel", "Practice Java collections and DSA."),
        ("Cloud Workshop", "workshop", 10, "10:00", "13:00",
         "College", "Introduction to cloud deployment."),
        ("Project Review", "meeting", 14, "15:00", "16:00",
         "Online", "Review Agentic RAG implementation."),
        ("Assignment Submission", "task", 20, "12:00", "13:00",
         "College Portal", "Submit the AI project assignment."),
        ("Career Workshop", "workshop", 25, "16:00", "18:00",
         "Seminar Hall", "Resume and interview preparation."),
    ]

    events = []
    for title, event_type, offset, start, end, location, description in sample:
        event_date = (today + timedelta(days=offset)).isoformat()
        events.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "type": event_type,
            "date": event_date,
            "start_time": start,
            "end_time": end,
            "location": location,
            "description": description,
        })
    return events


def initialize_schedule():
    if not os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
            json.dump(create_sample_schedule(), f, indent=2)


def load_schedule():
    initialize_schedule()
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_schedule(schedule):
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)


initialize_schedule()


# ============================================================
# 3. CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="schedule")


def event_to_text(event):
    return (
        f"Event ID: {event['id']}. "
        f"Title: {event['title']}. "
        f"Type: {event['type']}. "
        f"Date: {event['date']}. "
        f"Time: {event['start_time']} to {event['end_time']}. "
        f"Location: {event['location']}. "
        f"Description: {event['description']}."
    )


def rebuild_vector_database():
    global collection

    try:
        chroma_client.delete_collection(name="schedule")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name="schedule")
    schedule = load_schedule()

    if not schedule:
        return

    collection.add(
        ids=[e["id"] for e in schedule],
        documents=[event_to_text(e) for e in schedule],
        metadatas=[
            {
                "date": e["date"],
                "type": e["type"],
                "start_time": e["start_time"],
                "end_time": e["end_time"],
            }
            for e in schedule
        ],
    )


if collection.count() == 0:
    rebuild_vector_database()


# ============================================================
# 4. DATE / TIME HELPERS
# ============================================================

def normalize_date(value: Optional[str]):
    if not value:
        return None

    text = value.strip().lower()
    today = datetime.now().date()

    if text == "today":
        return today.isoformat()

    if text == "tomorrow":
        return (today + timedelta(days=1)).isoformat()

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }

    if text in weekdays:
        target = weekdays[text]
        days_ahead = (target - today.weekday()) % 7
        return (today + timedelta(days=days_ahead)).isoformat()

    for fmt in ("%Y-%m-%d", "%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt in ("%B %d", "%b %d"):
                parsed = parsed.replace(year=today.year)
            return parsed.date().isoformat()
        except ValueError:
            pass

    return text


def normalize_time(value: Optional[str]):
    if not value:
        return None

    text = value.strip().upper()

    for fmt in ("%I %p", "%I:%M %p", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).strftime("%H:%M")
        except ValueError:
            pass

    return text


def overlaps(event_start, event_end, requested_start, requested_end):
    if not requested_start and not requested_end:
        return True

    rs = requested_start or "00:00"
    re = requested_end or "23:59"
    return event_start < re and event_end > rs


# ============================================================
# 5. RAG RETRIEVAL
# ============================================================

def search_schedule(
    query: str,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_type: Optional[str] = None,
):
    schedule = load_schedule()
    normalized_date = normalize_date(date)
    normalized_start = normalize_time(start_time)
    normalized_end = normalize_time(end_time)

    # First apply exact structured filters.
    filtered = schedule

    if normalized_date:
        filtered = [e for e in filtered if e["date"] == normalized_date]

    if event_type:
        filtered = [
            e for e in filtered
            if e["type"].lower() == event_type.lower()
        ]

    if normalized_start or normalized_end:
        filtered = [
            e for e in filtered
            if overlaps(
                e["start_time"],
                e["end_time"],
                normalized_start,
                normalized_end,
            )
        ]

    # Then perform ChromaDB semantic retrieval.
    # If Chroma's embedding backend is unavailable, use the
    # structured-filtered data rather than crashing the app.
    try:
        count = collection.count()
        if count > 0:
            result = collection.query(
                query_texts=[query or "schedule"],
                n_results=min(10, count),
            )
            ids = result.get("ids", [[]])[0]

            by_id = {e["id"]: e for e in filtered}
            semantic = [by_id[i] for i in ids if i in by_id]

            if semantic:
                return semantic
    except Exception:
        pass

    return filtered


def format_events(events):
    if not events:
        return "No matching schedule events were found."

    parts = []
    for e in events:
        parts.append(
            f"ID: {e['id']}\n"
            f"Title: {e['title']}\n"
            f"Type: {e['type']}\n"
            f"Date: {e['date']}\n"
            f"Time: {e['start_time']} - {e['end_time']}\n"
            f"Location: {e['location']}\n"
            f"Description: {e['description']}"
        )
    return "\n\n".join(parts)


# ============================================================
# 6. REQUIRED TOOLS
# ============================================================

@tool
def get_schedule(
    query: str,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_type: Optional[str] = None,
) -> str:
    """Retrieve schedule information using date, time, type, and semantic query.

    Use this for questions about existing events, availability,
    meetings, workshops, tasks, appointments, today, tomorrow,
    weekdays, or finding an event before changing it.
    """
    events = search_schedule(
        query=query,
        date=date,
        start_time=start_time,
        end_time=end_time,
        event_type=event_type,
    )

    if not events and (
        "free" in query.lower()
        or "available" in query.lower()
    ):
        return "No event overlaps the requested time period. The user is free."

    return format_events(events)


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
    description: Optional[str] = None,
) -> str:
    """Add, update, or remove a schedule entry.

    operation must be add, update, or remove.
    For update/remove, event_id must identify the existing event.
    """

    operation = operation.lower().strip()
    schedule = load_schedule()

    if operation == "add":
        if not title or not date or not start_time:
            return "For add, title, date, and start_time are required."

        d = normalize_date(date)
        s = normalize_time(start_time)
        e = normalize_time(end_time)

        if not e:
            dt = datetime.strptime(s, "%H:%M") + timedelta(hours=1)
            e = dt.strftime("%H:%M")

        new_event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "type": event_type or "meeting",
            "date": d,
            "start_time": s,
            "end_time": e,
            "location": location or "Not specified",
            "description": description or "",
        }

        schedule.append(new_event)
        save_schedule(schedule)
        rebuild_vector_database()

        return "Event added successfully.\n\n" + event_to_text(new_event)

    if operation == "update":
        if not event_id:
            return "event_id is required for update."

        target = next((e for e in schedule if e["id"] == event_id), None)
        if not target:
            return "Event not found."

        if title is not None:
            target["title"] = title
        if event_type is not None:
            target["type"] = event_type
        if date is not None:
            target["date"] = normalize_date(date)
        if start_time is not None:
            target["start_time"] = normalize_time(start_time)
        if end_time is not None:
            target["end_time"] = normalize_time(end_time)
        if location is not None:
            target["location"] = location
        if description is not None:
            target["description"] = description

        save_schedule(schedule)
        rebuild_vector_database()

        return "Event updated successfully.\n\n" + event_to_text(target)

    if operation == "remove":
        if not event_id:
            return "event_id is required for removal."

        original = len(schedule)
        schedule = [e for e in schedule if e["id"] != event_id]

        if len(schedule) == original:
            return "Event not found."

        save_schedule(schedule)
        rebuild_vector_database()
        return "Event removed successfully."

    return "Invalid operation. Use add, update, or remove."


TOOLS = [get_schedule, update_schedule]
TOOL_MAP = {t.name: t for t in TOOLS}


# ============================================================
# 7. LANGGRAPH AGENT
# ============================================================

class ScheduleState(TypedDict):
    messages: List[BaseMessage]


SYSTEM_PROMPT = """
You are an Agentic RAG Schedule Assistant.

You manage the user's schedule for the next 30 days.

You have exactly two tools:
1. get_schedule
2. update_schedule

Rules:
- For questions about existing schedule information, call get_schedule.
- For adding, changing, or removing events, call update_schedule.
- For "What do I have scheduled tomorrow?", call get_schedule with date="tomorrow".
- For availability questions such as "Am I free Friday afternoon?",
  call get_schedule with the appropriate weekday and time range.
  Friday afternoon means approximately 12:00 to 17:00 unless the user
  specifies a different range.
- For "Add a meeting on August 15 at 3 PM", call update_schedule with
  operation="add". If no end time is given, use a one-hour duration.
- For "Move my meeting from 2 PM to 4 PM", FIRST call get_schedule to
  identify the event. AFTER receiving its result, call update_schedule
  with operation="update", the event_id from the result, and the new
  start_time. Do not guess an event_id.
- Never invent schedule events or IDs.
- After tools have supplied the required information, answer concisely.
"""


def agent_node(state: ScheduleState):
    response = llm.bind_tools(TOOLS).invoke(
        [HumanMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


def tools_node(state: ScheduleState):
    last = state["messages"][-1]
    tool_messages = []

    for call in getattr(last, "tool_calls", []):
        name = call["name"]
        args = call.get("args", {})

        if name not in TOOL_MAP:
            result = f"Unknown tool: {name}"
        else:
            try:
                result = TOOL_MAP[name].invoke(args)
            except Exception as exc:
                result = f"Tool error: {type(exc).__name__}: {exc}"

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
            )
        )

    return {"messages": tool_messages}


def route_after_agent(state: ScheduleState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", []):
        return "tools"
    return "done"


workflow = StateGraph(ScheduleState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "done": END,
    },
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()


# ============================================================
# 8. FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    description="30-day schedule assistant using Gemini, LangGraph and ChromaDB.",
    version="2.0.0",
)


def run_agent(task: str) -> str:
    result = graph.invoke(
        {"messages": [HumanMessage(content=task)]},
        config={"recursion_limit": 20},
    )

    messages = result.get("messages", [])

    # The final AI message is the last message without tool calls.
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", []):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    else:
                        texts.append(str(item))
                return "\n".join(texts)

    return "No response generated."


agent_runnable = RunnableLambda(
    lambda x: run_agent(x["task"] if isinstance(x, dict) else str(x))
)

add_routes(
    app,
    agent_runnable,
    path="/agent",
    playground_type="default",
)


class ScheduleRequest(BaseModel):
    task: str


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Agentic RAG Schedule Assistant is running",
        "playground": "/agent/playground/",
        "docs": "/docs",
        "test": "/test-agent",
        "schedule": "/schedule",
    }


@app.get("/schedule")
def view_schedule():
    return {"events": load_schedule()}


@app.get("/test-agent")
def test_agent():
    return {
        "task": "What do I have scheduled tomorrow?",
        "answer": run_agent("What do I have scheduled tomorrow?"),
    }


@app.post("/run")
def run_endpoint(request: ScheduleRequest):
    return {
        "task": request.task,
        "answer": run_agent(request.task),
    }


# ============================================================
# 9. START SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
