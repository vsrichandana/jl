import os
import json
import uuid
import shutil
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai


# ============================================================
# 1. CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

genai.configure(api_key=GOOGLE_API_KEY)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_FILE = "/tmp/schedule.json"
SEED_FILE     = os.path.join(BASE_DIR, "schedule.json")

# Copy seed data to writable /tmp on first boot
if not os.path.exists(SCHEDULE_FILE) and os.path.exists(SEED_FILE):
    shutil.copy(SEED_FILE, SCHEDULE_FILE)


# ============================================================
# 2. SCHEDULE DATA
# ============================================================

def create_sample_schedule() -> list:
    today = datetime.now().date()
    raw = [
        ("Team Standup",        "meeting",     0,  "10:00", "10:30", "Online",        "Daily project status meeting."),
        ("DSA Practice",        "task",        1,  "09:00", "10:00", "Hostel",         "Practice arrays and sliding window problems."),
        ("AI Workshop",         "workshop",    2,  "14:00", "17:00", "College Lab",    "Hands-on workshop about Agentic AI and RAG."),
        ("Project Meeting",     "meeting",     3,  "14:00", "15:00", "Online",         "Discuss major project progress."),
        ("Doctor Appointment",  "appointment", 5,  "11:00", "12:00", "City Clinic",    "Regular health check-up."),
        ("Java Practice",       "task",        7,  "18:00", "19:30", "Hostel",         "Practice Java collections and DSA."),
        ("Cloud Workshop",      "workshop",    10, "10:00", "13:00", "College",        "Introduction to cloud deployment."),
        ("Project Review",      "meeting",     14, "15:00", "16:00", "Online",         "Review Agentic RAG implementation."),
        ("Gym Session",         "task",        16, "07:00", "08:00", "Campus Gym",     "Morning workout session."),
        ("Resume Workshop",     "workshop",    18, "11:00", "13:00", "Seminar Hall",   "Build and refine your resume."),
        ("Mock Interview",      "appointment", 21, "10:00", "11:30", "Online",         "Practice technical interview questions."),
        ("Assignment Deadline", "task",        20, "12:00", "13:00", "College Portal", "Submit the AI project assignment."),
        ("Career Workshop",     "workshop",    25, "16:00", "18:00", "Seminar Hall",   "Resume and interview preparation."),
        ("Final Exam Prep",     "task",        28, "09:00", "12:00", "Library",        "Study for end-semester exams."),
        ("Team Lunch",          "meeting",     29, "13:00", "14:00", "Cafeteria",      "Team bonding lunch."),
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
        with open(SCHEDULE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return data
    with open(SCHEDULE_FILE) as f:
        return json.load(f)


def save_schedule(schedule: list):
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(schedule, f, indent=2)


# ============================================================
# 3. CHROMADB (in-memory)
# ============================================================

_chroma = chromadb.EphemeralClient()
_col    = _chroma.get_or_create_collection("schedule")


def event_to_text(e: dict) -> str:
    return (f"ID:{e['id']} Title:{e['title']} Type:{e['type']} "
            f"Date:{e['date']} Time:{e['start_time']}-{e['end_time']} "
            f"Location:{e['location']} Desc:{e['description']}")


def rebuild_chroma():
    global _col
    try:
        _chroma.delete_collection("schedule")
    except Exception:
        pass
    _col = _chroma.get_or_create_collection("schedule")
    data = load_schedule()
    if data:
        _col.add(
            ids=[e["id"] for e in data],
            documents=[event_to_text(e) for e in data],
            metadatas=[{"date": e["date"], "type": e["type"]} for e in data],
        )


rebuild_chroma()


# ============================================================
# 4. HELPERS
# ============================================================

def normalize_date(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    t = v.strip().lower()
    today = datetime.now().date()
    if t == "today":
        return today.isoformat()
    if t == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    days = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
            "friday":4,"saturday":5,"sunday":6}
    if t in days:
        ahead = (days[t] - today.weekday()) % 7 or 7
        return (today + timedelta(days=ahead)).isoformat()
    for fmt in ("%Y-%m-%d", "%B %d", "%b %d", "%B %d, %Y"):
        try:
            p = datetime.strptime(t, fmt)
            if fmt in ("%B %d", "%b %d"):
                p = p.replace(year=today.year)
            return p.date().isoformat()
        except ValueError:
            pass
    return v


def normalize_time(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    t = v.strip().upper().replace(".", "")
    for fmt in ("%I:%M %p", "%I %p", "%I%p", "%H:%M"):
        try:
            return datetime.strptime(t, fmt).strftime("%H:%M")
        except ValueError:
            pass
    return v


def fmt_events(events: list) -> str:
    if not events:
        return "No matching events found."
    out = []
    for e in events:
        out.append(
            f"• ID: {e['id']}\n"
            f"  {e['title']} ({e['type']})\n"
            f"  Date: {e['date']}  Time: {e['start_time']}–{e['end_time']}\n"
            f"  Location: {e['location']}\n"
            f"  {e['description']}"
        )
    return "\n\n".join(out)


# ============================================================
# 5. TOOL IMPLEMENTATIONS
# ============================================================

def tool_get_schedule(query: str, date: str = None, start_time: str = None,
                      end_time: str = None, event_type: str = None) -> str:
    data     = load_schedule()
    nd       = normalize_date(date)
    ns       = normalize_time(start_time)
    ne       = normalize_time(end_time)

    filtered = data
    if nd:
        filtered = [e for e in filtered if e["date"] == nd]
    if event_type:
        filtered = [e for e in filtered if e["type"].lower() == event_type.lower()]
    if ns or ne:
        rs = ns or "00:00"
        re = ne or "23:59"
        filtered = [e for e in filtered if e["start_time"] < re and e["end_time"] > rs]

    # semantic re-rank
    try:
        if _col.count() > 0:
            res = _col.query(query_texts=[query or "schedule"],
                             n_results=min(10, _col.count()))
            sem_ids = res["ids"][0]
            by_id   = {e["id"]: e for e in filtered}
            ranked  = [by_id[i] for i in sem_ids if i in by_id]
            if ranked:
                return fmt_events(ranked)
    except Exception:
        pass

    if not filtered and ("free" in query.lower() or "available" in query.lower()):
        return "You appear to be free during that time — no events found."
    return fmt_events(filtered)


def tool_update_schedule(operation: str, event_id: str = None, title: str = None,
                         event_type: str = None, date: str = None,
                         start_time: str = None, end_time: str = None,
                         location: str = None, description: str = None) -> str:
    op       = operation.lower().strip()
    schedule = load_schedule()

    if op == "add":
        if not title or not date or not start_time:
            return "title, date, and start_time are required to add an event."
        d = normalize_date(date)
        s = normalize_time(start_time)
        e = normalize_time(end_time)
        if not e:
            e = (datetime.strptime(s, "%H:%M") + timedelta(hours=1)).strftime("%H:%M")
        ev = {"id": str(uuid.uuid4()), "title": title,
              "type": event_type or "meeting", "date": d,
              "start_time": s, "end_time": e,
              "location": location or "TBD", "description": description or ""}
        schedule.append(ev)
        save_schedule(schedule)
        rebuild_chroma()
        return "Event added:\n\n" + fmt_events([ev])

    if op == "update":
        if not event_id:
            return "event_id is required for update."
        ev = next((x for x in schedule if x["id"] == event_id), None)
        if not ev:
            return f"No event found with id {event_id}."
        if title       is not None: ev["title"]       = title
        if event_type  is not None: ev["type"]        = event_type
        if date        is not None: ev["date"]        = normalize_date(date)
        if start_time  is not None: ev["start_time"]  = normalize_time(start_time)
        if end_time    is not None: ev["end_time"]    = normalize_time(end_time)
        if location    is not None: ev["location"]    = location
        if description is not None: ev["description"] = description
        save_schedule(schedule)
        rebuild_chroma()
        return "Event updated:\n\n" + fmt_events([ev])

    if op == "remove":
        if not event_id:
            return "event_id is required for remove."
        new = [x for x in schedule if x["id"] != event_id]
        if len(new) == len(schedule):
            return f"No event found with id {event_id}."
        save_schedule(new)
        rebuild_chroma()
        return "Event removed successfully."

    return "Invalid operation. Use add, update, or remove."


# ============================================================
# 6. GEMINI FUNCTION CALLING AGENT (no LangGraph)
# ============================================================

GET_SCHEDULE_DECL = genai.protos.FunctionDeclaration(
    name="get_schedule",
    description="Retrieve schedule events by date, time, type, or semantic query.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "query":      genai.protos.Schema(type=genai.protos.Type.STRING,
                          description="Natural language description of what to find."),
            "date":       genai.protos.Schema(type=genai.protos.Type.STRING,
                          description="Date: today, tomorrow, friday, 2026-08-15, August 15, etc."),
            "start_time": genai.protos.Schema(type=genai.protos.Type.STRING,
                          description="Filter start time e.g. 14:00 or 2 PM"),
            "end_time":   genai.protos.Schema(type=genai.protos.Type.STRING,
                          description="Filter end time e.g. 17:00 or 5 PM"),
            "event_type": genai.protos.Schema(type=genai.protos.Type.STRING,
                          description="meeting, task, workshop, or appointment"),
        },
        required=["query"],
    ),
)

UPDATE_SCHEDULE_DECL = genai.protos.FunctionDeclaration(
    name="update_schedule",
    description="Add, update, or remove a schedule entry.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "operation":   genai.protos.Schema(type=genai.protos.Type.STRING,
                           description="add, update, or remove"),
            "event_id":    genai.protos.Schema(type=genai.protos.Type.STRING,
                           description="ID of the event to update/remove (from get_schedule)"),
            "title":       genai.protos.Schema(type=genai.protos.Type.STRING),
            "event_type":  genai.protos.Schema(type=genai.protos.Type.STRING,
                           description="meeting, task, workshop, appointment"),
            "date":        genai.protos.Schema(type=genai.protos.Type.STRING),
            "start_time":  genai.protos.Schema(type=genai.protos.Type.STRING),
            "end_time":    genai.protos.Schema(type=genai.protos.Type.STRING),
            "location":    genai.protos.Schema(type=genai.protos.Type.STRING),
            "description": genai.protos.Schema(type=genai.protos.Type.STRING),
        },
        required=["operation"],
    ),
)

TOOLS_DECL = genai.protos.Tool(
    function_declarations=[GET_SCHEDULE_DECL, UPDATE_SCHEDULE_DECL]
)

SYSTEM_INSTRUCTION = (
    "You are an Agentic RAG Schedule Assistant managing a user's 30-day schedule.\n"
    "Today is " + datetime.now().date().isoformat() + ".\n\n"
    "Rules:\n"
    "- For ANY question about existing events → call get_schedule.\n"
    "- For availability (free/busy) → call get_schedule with the date and time range.\n"
    "  Friday afternoon = start_time=12:00, end_time=17:00.\n"
    "- To ADD an event → call update_schedule(operation=add, ...).\n"
    "- To MOVE/RESCHEDULE → FIRST call get_schedule to get the event_id, "
    "THEN call update_schedule(operation=update, event_id=..., new times).\n"
    "- To REMOVE → FIRST get_schedule, THEN update_schedule(operation=remove).\n"
    "- NEVER invent event IDs. Always get them from tool results.\n"
    "- Give concise, friendly answers after the tools respond."
)

TOOL_FN_MAP = {
    "get_schedule":    tool_get_schedule,
    "update_schedule": tool_update_schedule,
}


def run_agent(task: str) -> str:
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        tools=[TOOLS_DECL],
        system_instruction=SYSTEM_INSTRUCTION,
    )
    chat    = model.start_chat()
    history = []

    # Send user message
    response = chat.send_message(task)

    # Agentic loop — max 6 tool calls to prevent runaway
    for _ in range(6):
        part = response.candidates[0].content.parts[0]

        # If it's a function call, execute it and feed result back
        if part.function_call.name:
            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)

            if fn_name in TOOL_FN_MAP:
                try:
                    result = TOOL_FN_MAP[fn_name](**fn_args)
                except Exception as exc:
                    result = f"Tool error: {type(exc).__name__}: {exc}"
            else:
                result = f"Unknown tool: {fn_name}"

            # Feed the tool result back to the model
            response = chat.send_message(
                genai.protos.Content(
                    role="user",
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fn_name,
                            response={"result": result},
                        )
                    )]
                )
            )
        else:
            # Final text answer
            return part.text

    return "Could not generate a response."


# ============================================================
# 7. FASTAPI
# ============================================================

app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    version="4.0.0",
)


class RunRequest(BaseModel):
    task: str


@app.get("/agent/playground", response_class=HTMLResponse)
@app.get("/agent/playground/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def ui():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Schedule Assistant</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;
         min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:2rem}
    h1{font-size:2rem;margin-bottom:.4rem;color:#7dd3fc}
    .sub{color:#94a3b8;margin-bottom:1.8rem;font-size:.95rem}
    .card{background:#1e293b;border-radius:12px;padding:1.5rem;width:100%;max-width:700px;margin-bottom:1rem}
    textarea{width:100%;padding:.75rem;border-radius:8px;border:1px solid #334155;
             background:#0f172a;color:#e2e8f0;font-size:1rem;resize:vertical;min-height:80px}
    button{margin-top:.75rem;padding:.6rem 1.5rem;background:#3b82f6;color:#fff;
           border:none;border-radius:8px;font-size:1rem;cursor:pointer}
    button:hover{background:#2563eb}
    button:disabled{background:#475569;cursor:not-allowed}
    #resp{white-space:pre-wrap;background:#0f172a;padding:1rem;border-radius:8px;
          min-height:60px;color:#a3e635;font-size:.95rem;margin-top:1rem;line-height:1.6}
    .chips{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}
    .chip{background:#334155;padding:.4rem .9rem;border-radius:20px;font-size:.82rem;cursor:pointer}
    .chip:hover{background:#475569}
    a{color:#7dd3fc;text-decoration:none}a:hover{text-decoration:underline}
    .note{font-size:.78rem;color:#64748b;margin-top:.5rem}
  </style>
</head>
<body>
  <h1>📅 Schedule Assistant</h1>
  <p class="sub">Agentic RAG · Gemini · ChromaDB</p>
  <div class="card">
    <textarea id="inp" placeholder="Ask anything about your schedule…"></textarea>
    <div class="chips">
      <span class="chip" onclick="ask('What do I have scheduled tomorrow?')">Tomorrow?</span>
      <span class="chip" onclick="ask('Am I free Friday afternoon?')">Free Friday PM?</span>
      <span class="chip" onclick="ask('Show all my meetings this week')">This week meetings</span>
      <span class="chip" onclick="ask('Add a dentist appointment on August 20 at 10 AM')">Add event</span>
      <span class="chip" onclick="ask('Move my Project Meeting to 4 PM')">Reschedule</span>
      <span class="chip" onclick="ask('List all workshops')">All workshops</span>
    </div>
    <button id="btn" onclick="sendQuery()">Ask</button>
    <p class="note">First request may take ~30s if the server was idle.</p>
    <div id="resp">Response will appear here…</div>
  </div>
  <p style="font-size:.8rem;color:#475569">
    <a href="/docs">/docs</a> &nbsp;|&nbsp;
    <a href="/schedule">/schedule</a> &nbsp;|&nbsp;
    <a href="/health">/health</a>
  </p>
  <script>
    function ask(t){document.getElementById('inp').value=t;sendQuery()}
    async function sendQuery(){
      const task=document.getElementById('inp').value.trim();
      if(!task)return;
      const el=document.getElementById('resp');
      const btn=document.getElementById('btn');
      el.textContent='Thinking…';
      btn.disabled=true;
      const ctrl=new AbortController();
      const timer=setTimeout(()=>ctrl.abort(),180000);
      try{
        const r=await fetch('/run',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({task}),
          signal:ctrl.signal
        });
        clearTimeout(timer);
        if(!r.ok){el.textContent='Server error '+r.status+': '+await r.text();return}
        const d=await r.json();
        el.textContent=d.answer||JSON.stringify(d,null,2);
      }catch(e){
        clearTimeout(timer);
        el.textContent=e.name==='AbortError'
          ?'Timed out. Server may be waking up — please try again in a moment.'
          :'Error: '+e.message;
      }finally{btn.disabled=false}
    }
    document.getElementById('inp').addEventListener('keydown',e=>{
      if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendQuery()}
    });
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/run")
async def run_endpoint(request: RunRequest):
    loop   = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, run_agent, request.task)
    return {"task": request.task, "answer": answer}


@app.get("/schedule")
def view_schedule():
    return {"events": load_schedule()}


@app.get("/health")
def health():
    return {"status": "ok", "model": GEMINI_MODEL, "events": len(load_schedule())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
