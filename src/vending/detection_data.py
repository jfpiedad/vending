from vending.enums import AgeGroup, Weather


class FinalDetectionResults:
    age: int | None = None
    age_group: AgeGroup | None = None
    weather: Weather | None = None
    suggested_drinks: list[str] | None = None
