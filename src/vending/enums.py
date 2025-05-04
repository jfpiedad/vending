from enum import IntEnum, StrEnum, auto


class AgeGroup(StrEnum):
    CHILD = auto()
    TEEN = auto()
    ADULT = auto()
    SENIOR = auto()


class Weather(StrEnum):
    HOT = auto()
    COLD = auto()
    MODERATE = auto()


class MessageType(IntEnum):
    ALERT_MESSAGE = 0
    INITIAL = 1
    PROCESSING_USER = 2
    DETECTION_DATA = 3
    PROCESSING_DRINK = 4
    DRINK_READY = 5
