from datetime import datetime, timezone
from typing import Annotated

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from vending.enums import AgeGroup, MessageType, Weather


class BaseSchema(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class DetectionResults(BaseSchema):
    age: int
    age_group: AgeGroup
    weather: Weather
    suggested_drinks: list[str]


class Message(BaseSchema):
    message: DetectionResults | str
    message_type: MessageType


class TransactionBase(BaseSchema):
    age: int
    age_group: AgeGroup
    weather: Weather
    drink_bought: str
    timestamp: Annotated[
        datetime, Field(default_factory=lambda: datetime.now(timezone.utc))
    ]

    @field_validator("age_group", "weather", mode="before")
    @classmethod
    def convert_to_lowercase(cls, value: str) -> str:
        return value.lower()


class TransactionCreate(TransactionBase):
    pass


class Transaction(TransactionBase):
    id: Annotated[str, Field(validation_alias="_id")]

    @field_validator("id", mode="before")
    @classmethod
    def convert_objectid_to_str(cls, value: str | ObjectId) -> str:
        return str(value)
