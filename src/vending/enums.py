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
    INITIAL = 1
    DETECTION_DATA = 2
    PROCESSING_DRINK = 3
    DRINK_READY = 4
