import os
import traceback

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langserve import add_routes


# ============================================================
# GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

print("====================================")
print("GOOGLE API KEY CHECK")
print("====================================")

if GOOGLE_API_KEY:
    print("GOOGLE_API_KEY exists")
    print("Key length:", len(GOOGLE_API_KEY))
else:
    print("GOOGLE_API_KEY DOES NOT EXIST")


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# INPUT / OUTPUT
# ============================================================

class InputData(BaseModel):
    task: str


class OutputData(BaseModel):
    output: str


# ============================================================
# TEST GEMINI
# ============================================================

@app_placeholder = None


def test_gemini():

    try:

        print("Calling Gemini...")

        response = llm.invoke(
            "Say hello in one short sentence."
        )

        print("Gemini response:")
        print(response)

        return {
            "status": "success",
            "response": str(response.content)
        }

    except Exception as error:

        print("====================================")
        print("GEMINI ERROR")
        print("====================================")

        traceback.print_exc()

        return {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error)
        }


# ============================================================
# RUNNABLE
# ============================================================

def run_agent(data: InputData):

    try:

        print("Received:", data.task)

        response = llm.invoke(
            data.task
        )

        print("Gemini response:", response.content)

        return OutputData(
            output=str(response.content)
        )

    except Exception as error:

        print("====================================")
        print("RUN AGENT ERROR")
        print("====================================")

        traceback.print_exc()

        return OutputData(
            output=(
                "GEMINI ERROR: "
                + type(error).__name__
                + ": "
                + str(error)
            )
        )


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
    title="Gemini Render Diagnostic",
    version="1.0"
)


# ============================================================
# GEMINI DIRECT TEST
# ============================================================

@app.get("/test-gemini")
def test_gemini_endpoint():

    return test_gemini()


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
        "message": "Gemini diagnostic application"
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
