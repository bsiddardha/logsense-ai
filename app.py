from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse

from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from rag import ingest_logs
from rag import ask


app = FastAPI(title="LogSense AI")


templates = Jinja2Templates(directory="templates")


# ----------------------------
# Request Models
# ----------------------------

class UploadRequest(BaseModel):
    log: str


class QueryRequest(BaseModel):
    query: str


# ----------------------------
# Home Page
# ----------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# ----------------------------
# Upload Logs
# ----------------------------

@app.post("/upload")
async def upload_logs(data: UploadRequest):

    try:

        chunks = ingest_logs(data.log)

        return JSONResponse(
            {
                "success": True,
                "chunks": chunks
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e)
            }
        )


# ----------------------------
# Query Logs
# ----------------------------

@app.post("/query")
async def query_logs(data: QueryRequest):

    try:

        answer = ask(data.query)

        return JSONResponse(
            {
                "success": True,
                "answer": answer
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": str(e)
            }
        )