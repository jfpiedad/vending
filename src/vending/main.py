from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from vending.core import run_detection_livestream
from vending.database import initialize_db
from vending.schemas import Transaction
from vending.services import get_all_transactions_in_db
from vending.websocket import router as websocket_router

app = FastAPI(
    lifespan=initialize_db, swagger_ui_parameters={"operationsSorter": "method"}
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(websocket_router)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates("static")


@app.get("/", response_class=HTMLResponse)
def vending(request: Request) -> Any:
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/transaction-data", response_model=list[Transaction])
async def get_transaction_data(request: Request) -> Any:
    return await get_all_transactions_in_db(db=request.state.db)


@app.get("/camera-frames", response_class=StreamingResponse)
async def stream_camera_frames(request: Request) -> Any:
    return StreamingResponse(
        content=run_detection_livestream(request=request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
