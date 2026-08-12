import os
import traceback

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langserve import add_routes


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing."
    )

print("GOOGLE_API_KEY found.")


# ============================================================
# 2. GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. INPUT / OUTPUT
# ============================================================

class InputData(BaseModel):
    task: str


class OutputData(BaseModel):
    output: str


# ============================================================
# 4. DIRECT GEMINI TEST
# ============================================================

def call_gemini():

    try:

        print("Calling Gemini...")

        response = llm.invoke(
            "Say hello in one short sentence."
        )

        print("Gemini response:")
        print(response.content)

        return {
            "status": "success",
            "response": str(response.content)
        }

    except Exception as error:

        print("================================")
        print("GEMINI ERROR")
        print("================================")

        traceback.print_exc()

        return {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error)
        }


# ============================================================
# 5. GEMINI TEST ENDPOINT
# ============================================================

app = FastAPI(
    title="Gemini Render Diagnostic",
    version="1.0"
)


@app.get("/test-gemini")
def test_gemini():

    return call_gemini()


# ============================================================
# 6. LANGSERVE FUNCTION
# ============================================================

def run_agent(data: InputData) -> OutputData:

    try:

        print("Received request:")
        print(data.task)

        response = llm.invoke(
            data.task
        )

        print("Gemini response:")
        print(response.content)

        return OutputData(
            output=str(response.content)
        )

    except Exception as error:

        print("================================")
        print("RUN AGENT ERROR")
        print("================================")

        traceback.print_exc()

        return OutputData(
            output=(
                "Gemini error: "
                + type(error).__name__
                + ": "
                + str(error)
            )
        )


# ============================================================
# 7. CREATE RUNNABLE
# ============================================================

runnable = RunnableLambda(
    run_agent
).with_types(
    input_type=InputData,
    output_type=OutputData
)


# ============================================================
# 8. LANGSERVE
# ============================================================

add_routes(
    app,
    runnable,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 9. ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Gemini Render Diagnostic",
        "health": "/health",
        "test_gemini": "/test-gemini",
        "playground": "/agent/playground/",
        "docs": "/docs"
    }


# ============================================================
# 10. HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 11. START SERVER
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
