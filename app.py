import os

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda

from langserve import add_routes


# ============================================================
# API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing in Render Environment Variables."
    )


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# PLAYGROUND INPUT
# ============================================================

class InputData(BaseModel):
    task: str


class OutputData(BaseModel):
    output: str


# ============================================================
# TEST FUNCTION
# ============================================================

def run_agent(data: InputData) -> OutputData:

    print("Received request:", data.task)

    try:

        response = llm.invoke(
            data.task
        )

        print("Gemini response:", response.content)

        return OutputData(
            output=str(response.content)
        )

    except Exception as e:

        print(
            "GEMINI ERROR:",
            repr(e)
        )

        return OutputData(
            output=
            "Gemini Error: "
            + str(e)
        )


# ============================================================
# RUNNABLE
# ============================================================

runnable = RunnableLambda(
    run_agent
).with_types(
    input_type=InputData,
    output_type=OutputData
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Gemini Render Test",
    version="1.0"
)


# ============================================================
# LANGSERVE
# ============================================================

add_routes(
    app,
    runnable,
    path="/agent",
    playground_type="default"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Gemini Render application is running",
        "playground": "/agent/playground/",
        "health": "/health"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# START
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

