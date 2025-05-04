from vending.age_estimation.detect import AgeEstimator

VENDING_DRINKS = {
    "child": {
        "hot": "juice, flavored milk",
        "cold": "warm milk, hot chocolate",
        "moderate": "water, fruit smoothies",
    },
    "teen": {
        "hot": "soda, iced tea",
        "cold": "hot chocolate, tea",
        "moderate": "energy drinks, flavored water",
    },
    "adult": {
        "hot": "iced coffee, soft drinks",
        "cold": "coffee, tea",
        "moderate": "herbal tea, sparkling water",
    },
    "senior": {
        "hot": "water, iced tea",
        "cold": "herbal tea, warm water",
        "moderate": "green tea, fruit juice",
    },
}


AGE_DETECTOR = AgeEstimator()
