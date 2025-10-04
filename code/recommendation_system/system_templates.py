from typing import Dict, Union, List

system_templates: Dict[str, Union[List[str], Dict[str, List[str]]]] = {
    'general_welcome': [
        'System | Welcome to the Cambridge restaurant recommendation system!'
    ],
    'welcome': [
        "To narrow down suggestions please specify your preferences for the following field: cuisine (food type), "
        "price range and city area. If you don't care about some of those, just tell me!"
    ],
    'what?': [
        "Sorry, I didn't quite catch that.",
        "Can you rephrase that?",
        "I don't quite understand what you're saying. Please try again."
    ],
    'restaurant_info': {
        'food': [
            "{food} cuisine",
            "{food} food",
        ],
        'pricerange': [
            "in {pricerange} price range"
        ],
        'area': [
            "located in {area} area",
            "in {area} part of town"
        ],
        'romantic': {
            True: ["that is romantic"],
            False: ["that isn't romantic"]
        },
        'assignedseats': {
            True: ["that has assigned seats"],
            False: ["that doesn't have assigned seats"]
        },
        'children': {
            True: ["bringing children is a good idea"],
            False: ["bringing children isn't advised"]
        },
        'touristic': {
            True: ["that is touristic"],
            False: ["without a lot of tourists"]
        }
    },
    'confirm_selection': [
        "Great, you're interested in {restaurant_info}. Is that correct?",
        "Got it! You want {restaurant_info}. Is that right?",
        "Okay, you're looking for {restaurant_info}. Did I get that right?",
        "You prefer {restaurant_info}. Is that correct?",
        "You've selected {restaurant_info}. Is this correct?"
    ],
    'suggest_indirect_mappings': [
        "What would you like best out of {options}?",
        "May I suggest {options}? Which of these would you prefer?",
        "We have {options}. What would you like?"
    ],
    'negate_selection': [
        "I'm sorry, I didn't hear anything new! Could you repeat that?",
        "Oh no, I couldn't update your preferences based on that! Try telling me them again."
    ],
    'request_change': [
        "What about your given preferences would you like to change?"
    ],
    'request_area': [
        "What area would you like to dine in?",
        "Where in town do you want to eat?"
    ],
    'request_food': [
        "What cuisine are you into?",
        "What kind of restaurant are you looking for?"
    ],
    'request_pricerange': [
        "What kind of prices are you looking for? Expensive, moderate, cheap?",
        "What kind of price range did you have in mind? Cheap, moderate or something expensive?"
    ],
    'field_not_recognized': [
        "Sorry, I don't know about {field}. Could you rephrase?",
        "I haven't heard of {field}, please try again.",
        "I don't recognize {field}, can you tell me your preference again?"
    ],
    'ask_additional_reqs': [
        """Do you have any additional preferences regarding the following options? Use a comma to separate multiple requests.
        - Suitability for tourists or children.
        - Ideal for a romantic evening.
        - Availability of assigned seating."""
    ],
    'suggest_restaurant': [
        "How about {restaurantname}? It suits your preferences: {restaurant_info}. Is that a suitable option for you?",
        "Consider {restaurantname}! It fits well with your preferences: {restaurant_info}. Is this a satisfactory choice to you?"
    ],
    'no_restaurant_found': [
        "Sadly, I couldn't find (additional) {restaurant_info}. Could you let me know some altered preferences?"
    ],
    'propose_request_or_goodbye': [
        "Good to hear! Besides confirming your preferences you can also ask for any information on the restaurant such as a phone number, address, postal code etc. If you wish to end the conversation you can say goodbye."
    ],
    'end_conversation': [
        "Im glad you are happy with the choice we settled on and that everything is clear! I will end the conversation now."
    ],
    # Reasoning preferences
    'reasoning': {
        'direct': {
            # We know for sure that these restaurants conform to the requirements
            'touristic': [
                "It's cheap and has good food, so it attracts a lot of tourists."
            ],
            'not touristic': [
                "Romanian cuisine is unknown for most tourists and they prefer familiar food."
            ],
            'assignedseats': [
                "It's pretty busy, so the waiters generally decide where you sit."
            ],
            'not children': [
                "Nights out in this restaurant take a long time, so taking children is a bad idea."
            ],
            'not romantic': [
                "It's pretty busy, so it's no date material."
            ],
            'romantic': [
                "You can stay here for a long time, so it's pretty romantic."
            ]
        },
        'indirect': [
            # These restaurants probably refer to the requirements.
            "I can't guarantee {reasoning_preferences}, but I've at least excluded any restaurants that I know "
            "don't conform to that preference, so you're probably fine."
        ]
    },
    'answer_request': {
        'restaurantname': [
            "the restaurant is called {restaurantname}"
        ],
        'pricerange': [
            "their prices are considered to be {pricerange}"
        ],
        'area': [
            "they are positioned in the {area} part of town"
        ],
        'food': [
            "they are focused on serving {food} food"
        ],
        'quality': [
            "their food is rated as {quality}.",
            "according to reviews, their food is {quality}"
        ],
        'crowdedness': [
            "it's a {crowdedness} place"
        ],
        'lengthofstay': [
            "your stay is generally {lengthofstay} in duration"
        ],
        'phone': [
            "they are reachable on {phone}"
        ],
        'addr': [
            "it is located on {addr}"
        ],
        'postcode': [
            "the postcode is {postcode}"
        ]
    },
    'additional_reqconf': [
        "Do you have any other questions?",
        "Any more information you need?"
    ],
    'general_goodbye': [
        'System | Thank you for using the Cambridge restaurant recommendation system, we hope to see you again!\nExiting...'
    ]
}
