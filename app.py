import os
import json
import uuid
import shutil
from datetime import datetime, timedelta
from typing import TypedDict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. "
        "Add it in Render -> Environment Variables."
    )

# gemini-2.0-flash is available on all free Gemini API tiers.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# On Render the repo root is read-only after build.
# /tmp is always writable and survives for the life of the process.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_FILE = "/tmp/schedule.json"

# Pre-populate /tmp/schedule.json from the bundled seed file (first boot).
SEED_FILE = os.path.join(BASE_DIR, "schedule.json")
if not os.path.exists(SCHEDULE_FILE) and os.path.exists(SEED_FILE):
    shutil.copy(SEED_FILE, SCHEDULE_FILE)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# 2. SCHEDULE DATA (seed / load / save)
# ============================================================

def create_sample_schedule() -> list:
    today = datetime.now().date()
    raw = [
        ("Team Standup",         "meeting",     0,  "10:00", "10:30", "Online",         "Daily project status meeting."),
        ("DSA Practice",         "task",         1,  "09:00", "10:00", "Hostel",          "Practice arrays and sliding window problems."),
        ("AI Workshop",          "workshop",     2,  "14:00", "17:00", "College Lab",     "Hands-on workshop about Agentic AI and RAG."),
        ("Project Meeting",      "meeting",      3,  "14:00", "15:00", "Online",          "Discuss major project progress."),
        ("Doctor Appointment",   "appointment",  5,  "11:00", "12:00", "City Clinic",     "Regular health check-up."),
        ("Java Practice",        "task",         7,  "18:00", "19:30", "Hostel",          "Practice Java collections and DSA."),
        ("Cloud Workshop",       "workshop",     10, "10:00", "13:00", "College",         "Introduction to cloud deployment."),
        ("Project Review",       "meeting",      14, "15:00", "16:00", "Online",          "Review Agentic RAG implementation."),
        ("Gym Session",          "task",         16, "07:00", "08:00", "Campus Gym",      "Morning workout."),
        ("Resume Workshop",      "workshop",     18, "11:00", "13:00", "Seminar Hall",    "Build and refine your resume."),
        ("Mock Interview",       "appointment",  21, "10:00", "11:30", "Online",          "Practice technical interview questions."),
        ("Assignment Deadline",  "task",         20, "12:00", "13:00", "College Portal",  "Submit the AI project assignment."),
        ("Career Workshop",      "workshop",     25, "16:00", "18:00", "Seminar Hall",    "Resume and interview preparation."),
        ("Final Exam Prep",      "task",         28, "09:00", "12:00", "Library",         "Study for end-semester exams."),
        ("Team Lunch",           "meeting",      29, "13:00", "14:00", "Cafeteria",       "Team bonding lunch."),
    ]
    events = []
    for title, etype, offset, start, end, location, desc in raw:
        events.append({
            "id":          str(uuid.uuid4()),
            "title":       title,
            "type":        etype,
            "date":        (today + timedelta(days=offset)).isoformat(),
            "start_time":  start,
            "end_time":    end,
            "location":    location,
            "description": desc,
        })
    return events


def load_schedule() -> list:
    if not os.path.exists(SCHEDULE_FILE):
        data = create_sample_schedule()
        with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_schedule(schedule: list) -> None:
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)


# ============================================================
# 3. CHROMADB  (in-memory — works on any platform)
# ============================================================

# Use the built-in default embedding function (ONNX/MiniLM).
# No separate model download is needed — chromadb bundles it.
_ef = embedding_functions.DefaultEmbeddingFunction()

_chroma_client = chromadb.EphemeralClient()
_collection = _chroma_client.get_or_create_collection(
    name="schedule",
    embedding_function=_ef,
)


def event_to_text(event: dict) -> str:
    return (
        f"Event ID: {event['id']}. "
        f"Title: {event['title']}. "
        f"Type: {event['type']}. "
        f"Date: {event['date']}. "
        f"Time: {event['start_time']} to {event['end_time']}. "
        f"Location: {event['location']}. "
        f"Description: {event['description']}."
    )


def rebuild_vector_db() -> None:
    global _collection
    try:
        _chroma_client.delete_collection("schedule")
    except Exception:
        pass
    _collection = _chroma_client.get_or_create_collection(
        name="schedule",
        embedding_function=_ef,
    )
    schedule = load_schedule()
    if not schedule:
        return
    _collection.add(
        ids=[e["id"] for e in schedule],
        documents=[event_to_text(e) for e in schedule],
        metadatas=[
            {
                "date":       e["date"],
                "type":       e["type"],
                "start_time": e["start_time"],
                "end_time":   e["end_time"],
            }
            for e in schedule
        ],
    )


# Build on startup
rebuild_vector_db()


# ============================================================
# 4. DATE / TIME HELPERS
# ============================================================

def normalize_date(value: Optional[str]) -> Optional[str]:
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
        if days_ahead == 0:
            days_ahead = 7  # "next" occurrence
        return (today + timedelta(days=days_ahead)).isoformat()

    for fmt in ("%Y-%m-%d", "%B %d", "%b %d", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt in ("%B %d", "%b %d"):
                parsed = parsed.replace(year=today.year)
            return parsed.date().isoformat()
        except ValueError:
            pass

    return text  # pass through and let the LLM handle it


def normalize_time(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = value.strip().upper().replace(".", "")
    for fmt in ("%I %p", "%I:%M %p", "%H:%M", "%I%p"):
        try:
            return datetime.strptime(text, fmt).strftime("%H:%M")
        except ValueError:
            pass
    return text


def time_overlaps(
    ev_start: str,
    ev_end: str,
    req_start: Optional[str],
    req_end: Optional[str],
) -> bool:
    if not req_start and not req_end:
        return True
    rs = req_start or "00:00"
    re = req_end or "23:59"
    return ev_start < re and ev_end > rs


# ============================================================
# 5. RAG RETRIEVAL
# ============================================================

def search_schedule(
    query: str,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_type: Optional[str] = None,
) -> list:
    schedule = load_schedule()
    norm_date  = normalize_date(date)
    norm_start = normalize_time(start_time)
    norm_end   = normalize_time(end_time)

    # Structured filter first
    filtered = schedule
    if norm_date:
        filtered = [e for e in filtered if e["date"] == norm_date]
    if event_type:
        filtered = [e for e in filtered if e["type"].lower() == event_type.lower()]
    if norm_start or norm_end:
        filtered = [
            e for e in filtered
            if time_overlaps(e["start_time"], e["end_time"], norm_start, norm_end)
        ]

    # Semantic re-rank via ChromaDB (intersect with filtered)
    try:
        count = _collection.count()
        if count > 0:
            result = _collection.query(
                query_texts=[query or "schedule"],
                n_results=min(10, count),
            )
            semantic_ids = result.get("ids", [[]])[0]
            by_id = {e["id"]: e for e in filtered}
            reranked = [by_id[i] for i in semantic_ids if i in by_id]
            if reranked:
                return reranked
    except Exception:
        pass

    return filtered


def format_events(events: list) -> str:
    if not events:
        return "No matching schedule events found."
    parts = []
    for e in events:
        parts.append(
            f"• [{e['id']}] {e['title']} ({e['type']})\n"
            f"  Date: {e['date']}  |  Time: {e['start_time']} - {e['end_time']}\n"
            f"  Location: {e['location']}\n"
            f"  {e['description']}"
        )
    return "\n\n".join(parts)


# ============================================================
# 6. AGENT TOOLS
# ============================================================

@tool
def get_schedule(
    query: str,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_type: Optional[str] = None,
) -> str:
    """Retrieve schedule information.

    Use for:
    - Listing events on a specific date or weekday
    - Checking availability / free slots
    - Looking up an event before updating it
    - Any read-only query about the schedule

    Args:
        query:      Natural language description of what to find.
        date:       Date string — e.g. "today", "tomorrow", "friday",
                    "2026-08-15", "August 15".
        start_time: Filter to events starting at or after this time.
        end_time:   Filter to events ending at or before this time.
        event_type: One of meeting, task, workshop, appointment.
    """
    events = search_schedule(
        query=query,
        date=date,
        start_time=start_time,
        end_time=end_time,
        event_type=event_type,
    )

    if not events and ("free" in query.lower() or "available" in query.lower()):
        return "No events overlap that time window. The user appears to be free."

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

    Args:
        operation:  "add", "update", or "remove".
        event_id:   Required for update/remove — the ID from get_schedule.
        title:      Event title.
        event_type: meeting | task | workshop | appointment.
        date:       Date string — "2026-08-15", "August 15", "tomorrow", etc.
        start_time: e.g. "3 PM", "15:00", "3:30 PM".
        end_time:   e.g. "4 PM". Defaults to start + 1 hour when omitted.
        location:   Where the event takes place.
        description: Free-text notes.
    """
    op = operation.lower().strip()
    schedule = load_schedule()

    # ── ADD ──────────────────────────────────────────────────
    if op == "add":
        if not title or not date or not start_time:
            return "To add an event, title, date, and start_time are required."

        d = normalize_date(date)
        s = normalize_time(start_time)
        e = normalize_time(end_time)
        if not e:
            e = (datetime.strptime(s, "%H:%M") + timedelta(hours=1)).strftime("%H:%M")

        new_event = {
            "id":          str(uuid.uuid4()),
            "title":       title,
            "type":        event_type or "meeting",
            "date":        d,
            "start_time":  s,
            "end_time":    e,
            "location":    location or "Not specified",
            "description": description or "",
        }
        schedule.append(new_event)
        save_schedule(schedule)
        rebuild_vector_db()
        return "Event added successfully.\n\n" + format_events([new_event])

    # ── UPDATE ───────────────────────────────────────────────
    if op == "update":
        if not event_id:
            return "event_id is required for update."
        target = next((ev for ev in schedule if ev["id"] == event_id), None)
        if not target:
            return f"No event found with id '{event_id}'."

        if title       is not None: target["title"]       = title
        if event_type  is not None: target["type"]        = event_type
        if date        is not None: target["date"]        = normalize_date(date)
        if start_time  is not None: target["start_time"]  = normalize_time(start_time)
        if end_time    is not None: target["end_time"]    = normalize_time(end_time)
        if location    is not None: target["location"]    = location
        if description is not None: target["description"] = description

        save_schedule(schedule)
        rebuild_vector_db()
        return "Event updated successfully.\n\n" + format_events([target])

    # ── REMOVE ───────────────────────────────────────────────
    if op == "remove":
        if not event_id:
            return "event_id is required for remove."
        before = len(schedule)
        schedule = [ev for ev in schedule if ev["id"] != event_id]
        if len(schedule) == before:
            return f"No event found with id '{event_id}'."
        save_schedule(schedule)
        rebuild_vector_db()
        return "Event removed successfully."

    return "Invalid operation. Use 'add', 'update', or 'remove'."


TOOLS    = [get_schedule, update_schedule]
TOOL_MAP = {t.name: t for t in TOOLS}


# ============================================================
# 7. LANGGRAPH AGENT
# ============================================================

SYSTEM_PROMPT = """You are an Agentic RAG Schedule Assistant that manages a user's schedule.

Available tools:
1. get_schedule  – retrieve / search existing events
2. update_schedule – add, update, or remove events

Decision rules:
- Any question about what is scheduled → call get_schedule.
- Availability check ("Am I free Friday afternoon?") → call get_schedule
  with date="friday", start_time="12:00", end_time="17:00".
- Adding an event → call update_schedule(operation="add", ...).
- Moving / rescheduling an event → FIRST call get_schedule to get the
  event_id, THEN call update_schedule(operation="update", event_id=...).
- Removing an event → FIRST call get_schedule, THEN update_schedule(operation="remove").
- NEVER invent event IDs. Always obtain them from get_schedule results.
- After tools return data, give a concise, friendly answer.
- Today's date is """ + datetime.now().date().isoformat() + "."


class ScheduleState(TypedDict):
    messages: List[BaseMessage]


def agent_node(state: ScheduleState):
    llm_with_tools = llm.bind_tools(TOOLS)
    response = llm_with_tools.invoke(
        [HumanMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


def tools_node(state: ScheduleState):
    last = state["messages"][-1]
    results = []
    for call in getattr(last, "tool_calls", []):
        name = call["name"]
        args = call.get("args", {})
        if name not in TOOL_MAP:
            content = f"Unknown tool: {name}"
        else:
            try:
                content = TOOL_MAP[name].invoke(args)
            except Exception as exc:
                content = f"Tool error ({name}): {type(exc).__name__}: {exc}"
        results.append(
            ToolMessage(content=str(content), tool_call_id=call["id"])
        )
    return {"messages": results}


def route(state: ScheduleState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", []) else END


workflow = StateGraph(ScheduleState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()


# ============================================================
# 8. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    description=(
        "A 30-day schedule assistant powered by Gemini, LangGraph, and ChromaDB. "
        "POST /run with {\"task\": \"...\"}  or try the interactive UI at /chat."
    ),
    version="3.0.0",
)


def run_agent(task: str) -> str:
    """Invoke the LangGraph agent and return the final text response."""
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": 25},
        )
    except Exception as exc:
        return f"Agent error: {type(exc).__name__}: {exc}"

    for msg in reversed(result.get("messages", [])):
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", []):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    item["text"] if isinstance(item, dict) and "text" in item
                    else str(item)
                    for item in content
                )
    return "No response generated."


# ── REST endpoints ────────────────────────────────────────────

class RunRequest(BaseModel):
    task: str


@app.get("/agent/playground", response_class=HTMLResponse)
@app.get("/agent/playground/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def root():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>Schedule Assistant</title>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem; }
        h1 { font-size: 2rem; margin-bottom: 0.5rem; color: #7dd3fc; }
        p.sub { color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; width: 100%; max-width: 700px; margin-bottom: 1rem; }
        textarea { width: 100%; padding: 0.75rem; border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0; font-size: 1rem; resize: vertical; min-height: 80px; }
        button { margin-top: 0.75rem; padding: 0.6rem 1.5rem; background: #3b82f6; color: white; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; }
        button:hover { background: #2563eb; }
        #response { white-space: pre-wrap; background: #0f172a; padding: 1rem; border-radius: 8px; min-height: 60px; color: #a3e635; font-size: 0.95rem; margin-top: 1rem; }
        .examples { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
        .chip { background: #334155; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.82rem; cursor: pointer; }
        .chip:hover { background: #475569; }
        a.link { color: #7dd3fc; text-decoration: none; }
        a.link:hover { text-decoration: underline; }
      </style>
    </head>
    <body>
      <h1>📅 Schedule Assistant</h1>
      <p class="sub">Agentic RAG · Gemini · LangGraph · ChromaDB</p>
      <div class="card">
        <textarea id="inp" placeholder="Ask me anything about your schedule..."></textarea>
        <div class="examples">
          <span class="chip" onclick="ask('What do I have scheduled tomorrow?')">Tomorrow?</span>
          <span class="chip" onclick="ask('Am I free Friday afternoon?')">Free Friday PM?</span>
          <span class="chip" onclick="ask('Show all my meetings this week')">This week meetings</span>
          <span class="chip" onclick="ask('Add a dentist appointment on August 20 at 10 AM')">Add event</span>
          <span class="chip" onclick="ask('Move my Project Meeting to 4 PM')">Reschedule</span>
          <span class="chip" onclick="ask('List all workshops')">All workshops</span>
        </div>
        <button onclick="sendQuery()">Ask</button>
        <div id="response">Response will appear here…</div>
      </div>
      <p style="font-size:0.8rem; color:#475569;">
        API: <a class="link" href="/docs">/docs</a> &nbsp;|&nbsp;
        <a class="link" href="/schedule">/schedule</a> &nbsp;|&nbsp;
        POST <a class="link" href="/docs#/default/run_endpoint_run_post">/run</a>
      </p>
      <script>
        function ask(text) {
          document.getElementById('inp').value = text;
          sendQuery();
        }
        async function sendQuery() {
          const task = document.getElementById('inp').value.trim();
          if (!task) return;
          const el = document.getElementById('response');
          el.textContent = 'Thinking… (may take up to 30s on first request)';
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 120000);
          try {
            const res = await fetch('/run', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({task}),
              signal: controller.signal
            });
            clearTimeout(timer);
            if (!res.ok) {
              el.textContent = 'Server error ' + res.status + ': ' + await res.text();
              return;
            }
            const data = await res.json();
            el.textContent = data.answer || JSON.stringify(data);
          } catch (e) {
            clearTimeout(timer);
            if (e.name === 'AbortError') {
              el.textContent = 'Request timed out after 2 minutes. The server may be waking up — please try again.';
            } else {
              el.textContent = 'Error: ' + e.message;
            }
          }
        }
        document.getElementById('inp').addEventListener('keydown', e => {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/run")
async def run_endpoint(request: RunRequest):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, run_agent, request.task)
    except Exception as exc:
        import traceback
        answer = f"Agent error: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
    return {"task": request.task, "answer": answer}


@app.get("/schedule")
def view_schedule():
    return {"events": load_schedule()}


@app.get("/health")
def health():
    return {"status": "ok", "model": GEMINI_MODEL, "events": len(load_schedule())}


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
