import json
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from websockets import ConnectionClosed

from vending.detection_data import FinalDetectionResults
from vending.enums import MessageType
from vending.schemas import DetectionResults, Message
from vending.services import store_transaction_in_db
from vending.state import VendingState

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connection: WebSocket = None

    async def connect(self, websocket: WebSocket) -> None:
        if self.active_connection is not None:
            raise HTTPException(status_code=403, detail="Duplicate")

        await websocket.accept()
        self.active_connection = websocket

    def disconnect(self) -> None:
        self.active_connection = None

    async def send_data(self, data: str | dict[str, Any]) -> None:
        if isinstance(data, dict):
            await self.active_connection.send_json(data)
        elif isinstance(data, str):
            await self.active_connection.send_text(data)


manager = ConnectionManager()


@router.websocket("/ws/vending")
async def vending(websocket: WebSocket) -> Any:
    await manager.connect(websocket=websocket)

    try:
        while True:
            # Wait signal from client to begin detection pipeline.
            if not VendingState.currently_on:
                await websocket.receive_text()
                VendingState.currently_on = True

            # Initial message at the start of vending process.
            message = Message(message="", message_type=MessageType.INITIAL)
            await manager.send_data(message.model_dump(by_alias=True))

            # Wait until user decides to order.
            await websocket.receive_text()
            VendingState.currently_ordering = True

            # Edge Case: Order button is clicked but there is no user in front of the camera.
            # Return to initial state if there are no frames to be processed since there is no user.
            if VendingState.frames_count() < 10:
                VendingState.currently_ordering = False
                message = Message(
                    message="No user in front of the camera.",
                    message_type=MessageType.ALERT_MESSAGE,
                )
                await manager.send_data(message.model_dump(by_alias=True))
                continue

            # Wait until system is done processing user.
            message = Message(
                message="Processing user...", message_type=MessageType.PROCESSING_USER
            )
            await manager.send_data(message.model_dump(by_alias=True))

            # Wait until the system is finished processing the user.
            await VendingState.ready_to_vend.wait()

            if not VendingState.currently_ordering:
                # Reset back to initial state.
                VendingState.reset()
                message = Message(
                    message="No user in front of the camera.",
                    message_type=MessageType.ALERT_MESSAGE,
                )
                await manager.send_data(message.model_dump(by_alias=True))
                continue

            if not VendingState.currently_on:
                raise WebSocketDisconnect()

            detection_data = DetectionResults(
                age=FinalDetectionResults.age,
                age_group=FinalDetectionResults.age_group,
                weather=FinalDetectionResults.weather,
                suggested_drinks=FinalDetectionResults.suggested_drinks,
            )

            # Send the detection results data
            message = Message(
                message=detection_data, message_type=MessageType.DETECTION_DATA
            )
            await manager.send_data(message.model_dump(by_alias=True))

            # Wait for the user to choose a drink.
            # Tries to parse the received data as json. If it fails parsing, it means the user chose
            # to cancel, if it successfully parses, it means the user chose a drink.
            try:
                data = await websocket.receive_json()

                message = Message(
                    message=f"Processing {data['drinkBought']}...",
                    message_type=MessageType.PROCESSING_DRINK,
                )
                await manager.send_data(message.model_dump(by_alias=True))

                db = websocket.state.db
                await store_transaction_in_db(db=db, transaction_data=data)

                # Send a message after transaction is saved in the database.
                message = Message(
                    message=f"{data['drinkBought']} is done preparing.",
                    message_type=MessageType.DRINK_READY,
                )
                await manager.send_data(message.model_dump(by_alias=True))

                # Waiting for the user to confirm and go back to initial state.
                await websocket.receive_text()
            except json.JSONDecodeError:
                pass

            # Reset back to initial state.
            VendingState.reset()
    except (WebSocketDisconnect, ConnectionClosed):
        print("Websocket connection disconnected.")
        VendingState.reset()
        manager.disconnect()
