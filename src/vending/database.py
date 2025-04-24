from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from vending.config import get_settings


@asynccontextmanager
async def initialize_db(app: FastAPI) -> Any:
    cluster: AsyncIOMotorClient[dict[str, Any]] = AsyncIOMotorClient(
        host=get_settings().DB_CONNECTION_STRING
    )

    try:
        cluster.admin.command("ping")
        print("Successfully connected to the database.")
    except Exception as exc:
        print(exc)

    database = cluster[get_settings().DB_NAME]

    yield {"db": database}
