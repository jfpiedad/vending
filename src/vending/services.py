from datetime import timedelta
from typing import Any

from cachetools.func import ttl_cache
from httpx import Client
from motor.motor_asyncio import AsyncIOMotorDatabase

from vending.config import get_settings
from vending.enums import Weather
from vending.schemas import TransactionCreate


@ttl_cache(maxsize=128, ttl=timedelta(minutes=15).seconds)
def get_current_weather() -> Weather:
    url = get_settings().WEATHER_BASE_URL

    query_params = {
        "q": "10.293893, 123.897487",  #  (Latitude, Longitude) of Cebu City
        "key": get_settings().WEATHER_API_KEY,
    }

    with Client() as client:
        response = client.get(url=url, params=query_params)
        data = response.json()

    temperature = data["current"]["temp_c"]

    if temperature < 20:
        weather = Weather.COLD
    elif 20 <= temperature <= 30:
        weather = Weather.MODERATE
    else:
        weather = Weather.HOT

    return weather


async def store_transaction_in_db(
    db: AsyncIOMotorDatabase, transaction_data: dict[str, Any]
) -> None:
    transaction_details = TransactionCreate(**transaction_data)

    await db["transaction_data"].insert_one(document=transaction_details.model_dump())


async def get_all_transactions_in_db(db: AsyncIOMotorDatabase) -> list[dict[str, Any]]:
    return await db["transaction_data"].find({}).to_list(length=None)


if __name__ == "__main__":
    get_current_weather()
