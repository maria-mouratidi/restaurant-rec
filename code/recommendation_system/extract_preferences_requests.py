# Dependencies
import time
import Levenshtein as lev
import os
import sys
import re
import pandas as pd
import numpy as np
import os
import logging
import gensim.downloader
import string
from gensim.models import KeyedVectors
from typing import Tuple, Optional, List, Union, DefaultDict, Set
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)
sys.path.append('code')
from constants import COS_SIM_THRESHOLD
from field_mappings import (price_range_direct_mappings, area_direct_mappings, food_direct_mappings,
                            price_range_indirect_mappings, area_indirect_mappings, food_indirect_mappings)

# Config
logging.basicConfig(level=logging.INFO)

# Loading data
restaurant_info = pd.read_csv(os.path.join('data', 'restaurant_info.csv'))

w2v_path = os.path.join('code', 'recommendation_system', 'word2vec-google-news-300')
if not os.path.exists(w2v_path):
    w2v_model = gensim.downloader.load('word2vec-google-news-300')
    w2v_model.save(w2v_path)
w2v_model = KeyedVectors.load(w2v_path, mmap='r')


# Automatically extracting possible values for the preference fields.
price_ranges = set(restaurant_info['pricerange'].dropna())
areas = set(restaurant_info['area'].dropna())
foods = set(restaurant_info['food'].dropna())

logging.info(price_ranges)
logging.info(areas)
logging.info(foods)

price_range_vectors = {term: w2v_model[term] for term in price_ranges if term in w2v_model}
area_vectors = {term: w2v_model[term] for term in areas if term in w2v_model}
area_vectors['centre'] = w2v_model['center']
food_vectors = {term: w2v_model[term] for term in foods if term in w2v_model}
food_vectors['north american'] = w2v_model['american']
food_vectors['modern european'] = w2v_model['european']

# PREFERENCE FIELD EXTRACTION ==========================================================================================


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates the cosine similarity metric between the two supplied vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def levenshtein(extracted_field: str, all_field_options: Set[str]) -> Optional[str]:
    """The levenshtein function to correct input strings to any of their supplied options.
    There is a cap on edit distance to prevent things such as 'the' being changed to 'thai'.

    Args:
        extracted_field: the string of the field that was extracted by extract_preferences()
        all_field_options: the unique possible options to correct the input string to

    Returns:
        Either the closest match if the correction distance is less than max_lev, otherwise None
    """
    closest_match = min(all_field_options, key=lambda x: lev.distance(extracted_field, x))
    max_lev = 0.5 * min(len(extracted_field), len(closest_match))
    return closest_match if lev.distance(extracted_field, closest_match) < max_lev else None


def extract_preferences(utterance: str, use_w2v: bool = False, spec_field: Optional[str] = None) -> (
        defaultdict)[str, tuple[str, bool] | tuple[list[str], bool] | None]:
    """This function will extract preferences from a given input utterance.

    Args:
        utterance (str): utterance input classified as having dialogue act 'inform'
        use_w2v (bool): boolean depicting whether to use word2vec suggestions or not
        spec_field (optional, str): set this if the user was specifically asked for 1 field

    Returns:
        Dict[Tuple[str, bool]]: The return dictionary will have three keys, namely
        food, pricerange and area. Each of these keys can have the following 3 possible values:
                (term:str, True), Either levenshtein not necessary, or it was successful.
                (term:str, False): Levenshtein was applied but did not have any match.
                (term:List, False): multiple options for this preference
                None: The field was not found in the preferences
    """
    extracted_preferences: DefaultDict[str, Optional[Union[Tuple[str, bool], Tuple[List[str], bool]]]] \
        = defaultdict(lambda: None)
    utterance = utterance.lower()
    dontcare_pattern = r"(any(thing)?|(i )?(don'?t )?(care|mind)?( (about|remove)( the)?)?)"

    def extract_field(field: str, re_patterns: List[str], all_options_field: set, all_options_field_vectors: dict, field_mappings: Tuple):
        """In a number of different ways, try to fill the given field from preferences using the user's utterance.
        1. try to find a word in the string that exactly matches one of the field's options (exact matching without regex)
        2. try to find a number of optional keywords (direct mappings) in the utterance, that certainly point to a certain default option
        3. try to find a number of optional keywords (indirect mappings) in the utterance, that may point to a certain default option
        4. compare all words to the field's options and find the closest match with a Levenshtein edit distance <3 (if any)
        5. find a substring that may point to this field (e.g. words before "part of town") and
            a) use Levenshtein matching on that part once more
            b) find a "don't care" pattern, that may indicate that the user is indifferent about this field
            c) find similar words using a word2vec implementation and try to match those to the field's options

        Args:
            field: 'area', 'food' or 'pricerange'
            re_patterns: a pattern that matches on a substring of utterance that may contain information about this field
            all_options_field: list of all options for this field
            all_options_field_vectors: list of all cosine-similar words to each option for this field
            field_mappings: pair that contains direct and indirect mappings to each option for this field

        Returns:
            A single pair as described in the enclosing function's docstring
        """
        # EXACT MATCHING WITHOUT REGEX (saves computation time)
        for keyword in all_options_field:
            if keyword in utterance:
                extracted_preferences[field] = (keyword, True)
                logging.info(f'Exact match: {field}: {keyword}')
                return

        # DIRECT MAPPINGS - if one of these values is found, the key is automatically entered as a preference
        for key, value in field_mappings[0].items():
            for option in value:
                if option in utterance:
                    logging.info(f'Direct mapping: {field}: {option}')
                    extracted_preferences[field] = (key, True)
                    return
        # INDIRECT MAPPINGS - if one of these values is found, the user is asked whether the key is OK
        mapping_options = set()
        for key, value in field_mappings[1].items():
            for option in value:
                if option in utterance:
                    logging.info(f'Indirect mapping: {field}: {option}')
                    mapping_options.add(key)
        if mapping_options:
            extracted_preferences[field] = (list(mapping_options), False)
            return

        # LEVENSHTEIN MATCHING ON ALL KEYWORDS
        clean_words = (word.strip(string.punctuation) for word in utterance.split())
        filtered_words = [word for word in clean_words if len(word) > 3 and word.isalnum()]
        for word in filtered_words:
            closest_match = levenshtein(word, all_options_field)
            # LEVENSHTEIN <=3
            if closest_match:
                extracted_preferences[field] = (closest_match, True)
                logging.info(f'Levenshtein\'d match: {field}: {closest_match}')
                return

        # MATCHING ON REGEX RESULT
        for re_pattern in re_patterns:
            re_field = re.search(re_pattern, utterance, re.IGNORECASE)
            if re_field:
                extracted_field = re_field.group(0)
                logging.info(f'Regex match: {field}: {re_field.group(0)}')
                if extracted_field not in all_options_field:
                    closest_match = levenshtein(extracted_field, all_options_field)

                    # LEVENSHTEIN <=3 WITH OPTIONS
                    if closest_match:
                        extracted_preferences[field] = (closest_match, True)
                    # FIELD = DONT CARE PATTERN
                    elif re.search(dontcare_pattern, extracted_field, re.IGNORECASE):
                        extracted_preferences[field] = ('dont_care', True)
                    # WORD2VEC SUGGESTIONS
                    elif use_w2v and extracted_field in w2v_model:
                        term_vector = w2v_model[extracted_field]
                        vector_matches = [(key, cosine_similarity(term_vector, value)) for key, value in
                                          all_options_field_vectors.items() if cosine_similarity(term_vector, value) >= COS_SIM_THRESHOLD]
                        vector_matches.sort(key=lambda x: x[1], reverse=True)
                        extracted_preferences[field] = ([t[0] for t in vector_matches[:3]], False) if vector_matches else (extracted_field, False)
                    # EXTRACTED TEST FOR FIELD NOT MATCHABLE
                    else:
                        extracted_preferences[field] = (extracted_field, False)
                # EXACT MATCH WITH FIELD OPTION
                else:
                    extracted_preferences[field] = (extracted_field, True)
            return

    # Price ranges
    if not spec_field or spec_field == 'pricerange':
        if spec_field and re.search(dontcare_pattern, utterance):
            extracted_preferences[spec_field] = ('dont_care', True)
        extract_field(
            field='pricerange',
            re_patterns=['(\w+)(?=\s*((( about (the )?)?(pric(es?|ing)?( range)?))|(?:how cheap)))'],
            all_options_field=price_ranges,
            all_options_field_vectors=price_range_vectors,
            field_mappings=(price_range_direct_mappings, price_range_indirect_mappings)
        )

    # Areas
    if not spec_field or spec_field == 'area':
        if spec_field and re.search(dontcare_pattern, utterance):
            extracted_preferences[spec_field] = ('dont_care', True)
        extract_field(
            field='area',
            re_patterns=[r'(\w+)(?=\s*((( about (the )?)??:(part )?of town|area|city|cambridge|side)))'],
            all_options_field=areas,
            all_options_field_vectors=area_vectors,
            field_mappings=(area_direct_mappings, area_indirect_mappings)
        )

    # Food types
    if not spec_field or spec_field == 'food':
        if spec_field and re.search(dontcare_pattern, utterance):
            extracted_preferences[spec_field] = ('dont_care', True)
        extract_field(
            field='food',
            re_patterns = [
                r'food(?: (originating))? from (?:the |a )?(\w+)',
                r'\b(?!a\b|the\b|cheap\b|expensive\b|priced\b)(\w+)(?=\s*(?:food|cuisine|restaurant|bar|house|places?|options?|spots?|dishes|diners?))',
            ],
            all_options_field=foods,
            all_options_field_vectors=food_vectors,
            field_mappings=(food_direct_mappings, food_indirect_mappings)
        )

    logging.debug(f'Found preferences: {dict(extracted_preferences)}')
    return extracted_preferences


def update_preferences(old_prefs: dict, new_prefs: dict) -> dict:
    """Given the current set of preferences, update those that are correctly filled in new_prefs.

    Args:
        old_prefs: the current preferences dictionary
        new_prefs: a new preferences dictionary

    Returns:
        old_prefs, but all fields that are different in new_prefs updated to the new_prefs value
    """
    # Update if the new preferences contain new information for a field
    for field in ['food', 'area', 'pricerange']:
        if new_prefs[field] and new_prefs[field][1]:
            old_prefs[field] = new_prefs[field]

    logging.debug(f'Preferences updated to {dict(old_prefs)}')
    return old_prefs


# REQUEST FIELD EXTRACTION =============================================================================================

def extract_request_fields(utterance: str, confirm: bool = False) -> set[str]:
    """Given a request utterance from the user, find out which fields are requested.
    This function is also used to handle 'confirm' acts, which first requires the term to be confirmed
    to be recognized and changed to its corresponding keyword.

    Args:
        utterance: input utterance from which to extract fields
        confirm: boolean specifying whether to first parse term to keyword in case of 'confirm' act

    Returns:
        In both cases it will return a set of keywords for which to look up and communicate restaurant info.
    """
    keywords = {
        'restaurantname': r'name|called',
        'pricerange': r'price?|expensive|cheap|money',
        'area': r'where|area|part\s?(of\s*(town|city|cambridge))?',
        'food': r'(what|type(\s*of)?)\s*?food|cuisine',
        'quality': r'(how\s*)?good|rating|quality',
        'crowdedness': r'(how\s*)?busy|crowded|quiet',
        'lengthofstay': r'how long|hours|time|stay',
        'phone': r'phone|number',
        'addr': r'add?ress?|street',
        'postcode': r'(post(al)?(\s*code)?|code)'
    }

    if confirm:
        return {key for key, value in extract_preferences(utterance).items() if
                value is not None and isinstance(value, tuple) and value[1]}
    else:
        return {keyword for keyword, pattern in keywords.items() if re.search(pattern, utterance)}


# PREFERENCES FOR REASONING ============================================================================================

def extract_add_requirements(utterance: str) -> dict:
    """Will extract the reasoning requirements from a given input string.
    A negate pattern is used to check whether a certain sub-sentence specifically does NOT want the preference.

    Args:
        utterance: the full user input

    Returns:
        a preferences dictionary specifically for touristic, assignedseats, children and romantic values
    """
    preference_patterns = {
        'touristic': re.compile(r'touris(tic|ts)', re.IGNORECASE),
        'assignedseats': re.compile(r'assigned( seats)?|choose.*seats', re.IGNORECASE),
        'children': re.compile(r'child(ren)?|kid(s)?|daughter|son|baby', re.IGNORECASE),
        'romantic': re.compile(r'roman(tic|ce)|[Vv]alentine', re.IGNORECASE)
    }
    negate_pattern = re.compile(r"\b(don'?t|not?|without|do not)\b", re.IGNORECASE)
    preferences: DefaultDict[str, bool] = defaultdict(lambda: False)

    for sub_sentence in utterance.split(','):
        for preference, pattern in preference_patterns.items():
            if re.search(pattern, sub_sentence):
                preferences[preference] = not bool(re.search(negate_pattern, sub_sentence))
    return preferences

# TESTING ==============================================================================================================

if __name__ == '__main__':
    """Used for testing the extract_preferences function"""
    preference_test_sentences = [
        ("I'm looking for world food",
         {'food': ('international', True)}),
        ("I want a restaurant that serves world food",
         {'food': ('international', True)}),
        ("I want a restaurant serving Swedish food",
         {'food': (['european'], False)}),
        ("I'm looking for a restaurant in the center",
         {'area': ('centre', True)}),
        ("I would like a cheap restaurant in the west part of town",
         {'pricerange': ('cheap', True), 'area': ('west', True)}),
        ("I'm looking for a moderately priced restaurant in the west part of town",
         {'pricerange': ('moderate', True), 'area': ('west', True)}),
        ("I'm looking for a restaurant in any area that serves Tuscan food",
         {'area': ('dont_care', True), 'food': ('tuscan', True)}),
        ("Can I have an expensive restaurant",
         {'pricerange': ('expensive', True)}),
        ("I'm looking for an expensive restaurant and it should serve international food",
         {'pricerange': ('expensive', True), 'food': ('international', True)}),
        ("I need a Cuban restaurant that is moderately priced",
         {'pricerange': ('moderate', True), 'food': ('cuban', True)}),
        ("I'm looking for a moderately priced restaurant with Catalan food",
         {'pricerange': ('moderate', True), 'food': ('catalan', True)}),
        ("What is a cheap restaurant in the south part of town",
         {'pricerange': ('cheap', True), 'area': ('south', True)}),
        ("What about Chinese food",
         {'food': ('chinese', True)}),
        ("I wanna find a cheap restaurant",
         {'pricerange': ('cheap', True)}),
        ("I'm looking for Persian food please",
         {'food': ('persian', True)}),
        ("Find a Cuban restaurant in the center",
         {'area': ('centre', True), 'food': ('cuban', True)}),
    ]

    correct = 0
    for sentence, expected_output in preference_test_sentences:
        preferences = extract_preferences(sentence)
        preferences = dict(preferences)  # type: ignore
        try:
            assert preferences == expected_output
            correct += 1
        except:
            print(f"For input '{sentence}',\n\texpected {expected_output}\n\tbut got {preferences}", '\n')
            continue

    print(f"Accuracy: {correct / len(preference_test_sentences) * 100}%")

    # Console testing ===================================
    while True:
        sentence = input("Please enter a test sentence (or '\quit' to exit): ")

        if sentence.lower() == '\quit':
            break

        preferences = extract_preferences(sentence)
        print(f"'{sentence}': {dict(preferences)}")
        time.sleep(1)
