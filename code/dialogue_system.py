# Dependencies and config ======================================================'
import os
import sys
import time
import re
import random
import pandas as pd
import logging
import pyttsx3
import speech_recognition as sr
import keyboard
import requests
import hashlib
from collections import defaultdict
from typing import Optional

# Avoid logging output of imports
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame import mixer
mixer.init()

# User dependencies
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.append('code')
sys.path.append((os.path.join('code', 'rule_based')))
sys.path.append(os.path.join('code', 'recommendation_system'))
from constants import (SYSTEMS, COMMANDS, INFERENCE_RULES, DF_DIALOGUE_ACTS, DF_DIALOGUE_ACTS_DD, PREFERENCE_NAMES)
from majority_class.system_majority_class import classify_majority_class, evaluate_majority_class
from rule_based.system_rule_based import classify_rule_based, evaluate_rule_based
from decision_tree.system_decision_tree import classify_decision_tree, evaluate_decision_tree, dt_load_mod_vec_enc
from feed_forward.system_feed_forward import classify_feed_forward, evaluate_feed_forward, ff_load_mod_vec_enc
from recommendation_system.system_templates import system_templates
from recommendation_system.extract_preferences_requests import (extract_preferences, update_preferences,
                                                                extract_request_fields, extract_add_requirements)


class Dialogue:
    def __init__(self, condition=None):
        """Full initiation of the Dialogue class"""
        self.active_system = 'feed_forward-dd'
        self.model_deps = ff_load_mod_vec_enc('deduplicated')
        self.system_call = classify_feed_forward
        self.evaluation_call = evaluate_feed_forward
        self.condition = condition

        if self.condition in 'Aa':
            self.n_of_options = 2
        elif self.condition in 'Bb':
            self.n_of_options = 5
        else:
            raise Exception('Condition should be either "A" or "B"!')

        #print(f'Starting experiment with condition {condition}\n')
        print(f'Using default classification system: {self.active_system}\n')
        print(COMMANDS)

        self.preferences = defaultdict(lambda: None)
        self.reasoning_preferences = defaultdict(lambda: None)

        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        self.restaurant_catalogue = pd.read_csv(os.path.join('data', 'restaurant_info.csv'))
        self.restaurant_catalogue['suggested'] = False
        self.current_suggestion_info = None

        print(random.choice(system_templates['general_welcome']))
        self.current_state = 's1_welcomed'
        self.system_utterance = self.utter('welcome')
        self.dialogue_act = None
        self.user_input = None

        self.tts_engine = None
        self.w2v_on = True
        self.tts_on = False
        self.stt_on = False
        self.debug_on = False

    def reset_conversation(self):
        """Fully resets the dialogue state by calling the init function."""
        #condition = 'A' if self.n_of_options == 5 else 'B'
        self.__init__(condition=self.condition)

    def end_conversation(self):
        """Ends the current Dialogue and exits the program without errors."""
        print(random.choice(system_templates['general_goodbye']))
        sys.exit(0)

    # USER INTERACTION =================================================================================================

    def classification_test_mode(self):
        """Enables a mode in which the currently activated act classification system can be tested."""
        print(f'Testing mode enabled, active system: {self.active_system}\n')

        while True:
            print(f'{self.active_system.upper()} | Enter an utterance to be classified. Write \\stop_test to exit.')
            user_input = input('You | ').lower()

            if user_input == r'\stop_test':
                print()
                break

            print(f'{self.active_system.upper()} | Act = {self.system_call(user_input, *self.model_deps)}\n')

    def record_user_input(self):
        """ Handles (re)-recording and speech to text conversion based on buttons.

        Returns:
            str containing user input
        """
        exit_flag = False

        def on_escape():
            """Nested helper function that exits STT based on Escape event """
            nonlocal exit_flag
            print("\nSTT | Escape pressed, speech-to-text toggled off.")
            exit_flag = True

        keyboard.on_press_key("esc", on_escape)

        while True:
            if exit_flag:
                keyboard.unhook_all()
                return None

            with self.mic as source:
                print("STT | Calibrating microphone (3s). Please wait...")
                self.recognizer.adjust_for_ambient_noise(source, duration=3)

            print("STT | Press 's' to start recording and 'esc' to exit speech-to-text: ", end='')
            while True:
                if exit_flag:
                    keyboard.unhook_all()
                    return None
                if keyboard.is_pressed('s'):
                    break
                time.sleep(0.1)

            with self.mic as source:
                print("\nSTT | Listening. Recording will end automatically. (12s automatic max)")
                audio_data = self.recognizer.listen(source, timeout=3, phrase_time_limit=12)
                keyboard.unhook_all()

            try:
                print(f"\nSTT | Processing... ")
                recognized_text = self.recognizer.recognize_google(audio_data)
                print(f"STT | Recognised speech: {recognized_text}")
            except sr.UnknownValueError:
                print("\nSTT | Could not understand audio")
                continue
            except sr.RequestError as e:
                print(f"\nSTT | Could not request results; {e}")
                continue

            print("STT | Press 'a' to accept or 'r' to re-record: ", end='')
            while True:
                if keyboard.is_pressed('a'):
                    return recognized_text
                elif keyboard.is_pressed('r'):
                    time.sleep(0.5)
                    print('\n')
                    break

    def redetermine_system(self) -> None:
        """ User interaction function to determine the desired classification system.

        Returns:
            void, but fills self.active_system, self.system_call, and self.evaluation_call
        """
        print(f'Available dialogue act classification systems:\n\t{", ".join(SYSTEMS)}')

        while not self.active_system:
            desired_system = input('Desired classification system: ')

            # Exiting application
            if desired_system.lower() == r'\quit':
                print(system_templates['general_goodbye'])
                return

            if desired_system not in SYSTEMS:
                print(f'Classification system not found! Try again please.')
                continue

            self.active_system = desired_system
            ds_split = desired_system.split('-')

            match ds_split[0]:
                case 'majority_class':
                    self.system_call = classify_majority_class
                    self.evaluation_call = evaluate_majority_class
                case 'rule_based':
                    self.system_call = classify_rule_based
                    self.evaluation_call = evaluate_rule_based
                case 'decision_tree':
                    self.model_deps = dt_load_mod_vec_enc(
                        'deduplicated') if ds_split[1] == 'dd' else dt_load_mod_vec_enc('full')
                    self.system_call = classify_decision_tree
                    self.evaluation_call = evaluate_decision_tree
                case 'feed_forward':
                    self.model_deps = ff_load_mod_vec_enc(
                        'deduplicated') if ds_split[1] == 'dd' else ff_load_mod_vec_enc('full')
                    self.system_call = classify_feed_forward
                    self.evaluation_call = evaluate_feed_forward

        print(f'Classification system Chosen: {self.active_system}\n')

    # STATE TRANSITIONING UTILS ========================================================================================

    def utter(self, template_name: str, fill: Optional[dict] = None, request: bool = False, name: Optional[str] = None) -> str:
        """ Given a key for a template from system_templates, create a system_utterance out of it.

        Args:
            template_name: the name of the corresponding template utterance in system_templates.py
            fill: any format arguments to fill a template with
            request: whether it is a request response or not
            name: the restaurant name

        Returns:
            A string that the system can use as output
        """
        template_mapping = system_templates[template_name]
        template = system_templates[template_name] if isinstance(template_mapping, dict) else random.choice(template_mapping)

        if template_name == 'what?':
            return '' if any(template in self.system_utterance for template in system_templates['what?']) else template  # type: ignore
        elif fill:
            restaurant_info_parts = []

            if not request:
                # First utter the food preference, for sentence structure
                food_value = fill['food']
                value_str = 'any' if food_value[0] == 'dont_care' else food_value[0]
                template_str = random.choice(system_templates['restaurant_info']['food'])
                restaurant_info_parts.append(template_str.format(**{'food': value_str}))

            # Then utter all other preferences
            for pref_key, pref_value in fill.items():
                if request and pref_value:
                    template_str = random.choice(template[pref_key])
                    restaurant_info_parts.append(template_str.format(**{pref_key: pref_value}))
                elif pref_value and pref_key in list(zip(*INFERENCE_RULES))[0]:
                    template_str = random.choice(system_templates['restaurant_info'][pref_key][pref_value])
                    restaurant_info_parts.append(template_str.format(**{pref_key: pref_value}))
                elif pref_value and pref_key != 'food':
                    value_str = 'any' if pref_value[0] == 'dont_care' else 'the ' + pref_value[0]
                    template_str = random.choice(system_templates['restaurant_info'][pref_key])
                    restaurant_info_parts.append(template_str.format(**{pref_key: value_str}))

            restaurant_info = ', '.join(restaurant_info_parts).strip()
            if request:
                restaurant_info = restaurant_info[0].upper() + restaurant_info[1:] + '.'

            return restaurant_info if request else template.format(restaurant_info=restaurant_info, restaurantname=name)  # type: ignore
        else:
            return template  # type: ignore

    def handle_first_prefs(self):
        """ Used in s1, these are the commands necessary to initially fill preferences. Reads the user input, fills
        self.preferences with the correctly filled fields, and gives error messages for fields that weren't filled
        correctly.

        Returns:
            void, but fills self.preferences
        """
        new_prefs = extract_preferences(self.user_input, use_w2v=self.w2v_on)

        if not new_prefs:
            self.system_utterance = f"{self.utter('what?')}\n\t{self.system_utterance}"
        if all(pref[1] for pref in new_prefs.values() if pref is not None):
            # All fields that the user mentioned have been successfully filled
            self.preferences = new_prefs
            self.check_if_prefs_filled()
        else:
            # Some field was filled unsuccessfully, because either Levenshtein > 3 or there's an indirect mapping
            # First fill the fields that did work
            self.preferences = update_preferences(self.preferences, new_prefs)

            # Then give an error for one that didn't
            for field in [f for f in new_prefs if new_prefs[f] is not None]:
                field_content, success = new_prefs[field]
                if not success:
                    if type(field_content) == str:
                        # This field had a Levenshtein edit distance > 3
                        self.system_utterance = self.utter('field_not_recognized').format(field=field_content)
                        self.current_state = f's6_requested_field_{field}'
                    elif type(field_content) == list:
                        # This field was not recognized but has an indirect mapping
                        self.suggest_indirect_mappings(field, field_content)

    def handle_prefs_update(self):
        """Used in s2 and s7, this function takes the user_input and updates any changed fields when the user wants to
        change their previously filled preferences. If no fields change, it specifically asks what the user wants to
        change.

        Returns:
            void, but fills self.preferences, self.system_utterance, and self.current_state
        """
        old_prefs = self.preferences.copy()
        new_prefs = extract_preferences(self.user_input, self.w2v_on)
        self.preferences = update_preferences(self.preferences, new_prefs)
        if any(not pref[1] for pref in new_prefs.values() if pref is not None):
            # Some field was filled unsuccessfully, because either Levenshtein > 3 or there's an indirect mapping
            # First fill the fields that did work
            self.preferences = update_preferences(self.preferences, new_prefs)

            # Then give an error for one that didn't
            for field in [f for f in new_prefs if new_prefs[f] is not None]:
                field_content, success = new_prefs[field]
                if not success:
                    if type(field_content) == str:
                        # This field had a Levenshtein edit distance > 3
                        self.system_utterance = self.utter('field_not_recognized').format(field=field_content)
                        self.current_state = f's6_requested_field_{field}'
                    elif type(field_content) == list:
                        # This field was not recognized but has an indirect mapping
                        self.suggest_indirect_mappings(field, field_content)
        elif old_prefs == self.preferences:
            # Nothing changed, ask for information
            self.system_utterance = self.utter('negate_selection')
            self.current_state = 's7_queried_for_change'
        else:   # all(pref[1] for pref in new_prefs.values() if pref is not None):
            # All fields that the user mentioned have been successfully filled
            self.preferences = update_preferences(self.preferences, new_prefs)
            self.system_utterance = self.utter('confirm_selection', fill=self.preferences)
            self.current_state = 's2_confirmed_selection'

    def suggest_indirect_mappings(self, field, field_options):
        """ Suggest either 2 or 5 options for this category that are similar to the given input """
        if self.n_of_options:
            # In the experiment, give only 2 options if we are in that condition
            fewer_options = set()
            for i in range(self.n_of_options):
                choice = random.choice(field_options)
                field_options.remove(choice)
                fewer_options.add(choice)
            field_options = fewer_options
        options = ', '.join(field_options)
        self.system_utterance = self.utter('suggest_indirect_mappings').format(options=options)
        self.current_state = f's6_requested_field_{field}'

    def check_if_prefs_filled(self):
        """ Check whether self.preferences contains a value for each of the three fields. If so, go to
        confirm_selection. If not, ask for a missing one.

        Returns:
            void, but fills self.system_utterance and self.current_state
        """
        unmentioned_fields = set(PREFERENCE_NAMES) - set(self.preferences.keys())
        if not unmentioned_fields:
            logging.debug(f'Preferences filled to {dict(self.preferences)}')
            self.system_utterance = self.utter('confirm_selection', fill=self.preferences)
            self.current_state = 's2_confirmed_selection'
        else:
            # Not all fields have been mentioned yet, ask for a specific one
            spec_field = random.choice(list(unmentioned_fields))
            self.system_utterance = self.utter(f'request_{spec_field}')
            self.current_state = f's6_requested_field_{spec_field}'

    def pick_restaurant_option(self):
        """ Get the information for a random restaurant that satisfies the user's requirements.

        Returns:
            void, but fills self.current_suggestion_info
        """
        current_catalogue = self.restaurant_catalogue[~self.restaurant_catalogue['suggested']].copy()
        current_catalogue['satisfies_add_req'] = 'maybe'

        # Update catalogue according to preferences
        for key, value in self.preferences.items():
            if value and value[0] != 'dont_care':
                current_catalogue = current_catalogue[current_catalogue[key] == value[0]]

        # Update catalogue according to additional preferences
        if self.reasoning_preferences:
            for field, preferred_truth in self.reasoning_preferences.items():
                # Find all rules that have this preference as a consequent
                # and reverse its order for priority
                relevant_rules = list(reversed([(c, r, it) for c, r, it in INFERENCE_RULES
                                                if c == field]))

                for consequent, rule, inferred_truth in relevant_rules:
                    if preferred_truth == inferred_truth:
                        # We can directly infer that these restaurants conform to the user's preferences
                        current_catalogue.loc[lambda row: rule(row), 'satisfies_add_req'] = 'yes'
                    else:
                        # We can't directly infer, but we can at least remove the restaurants that we know don't conform
                        current_catalogue.loc[lambda row: rule(row), 'satisfies_add_req'] = 'no'

            if not current_catalogue[current_catalogue['satisfies_add_req'] == 'yes'].empty:
                current_catalogue = current_catalogue[current_catalogue['satisfies_add_req'] == 'yes']
            else:
                current_catalogue = current_catalogue[current_catalogue['satisfies_add_req'] != 'no']

        if len(current_catalogue) >= 1:
            random_index = random.choice(current_catalogue.index)
            self.restaurant_catalogue.loc[random_index, 'suggested'] = True
            self.current_suggestion_info = current_catalogue.loc[random_index].to_dict()
        else:
            self.current_suggestion_info = {}

    def suggest_candidate(self):
        """ Sequence of actions for finding a restaurant that conforms to all preferences, then either suggesting that
        one or letting the user know there aren't any options.

        Returns:
            void, but fills self.current_suggestion_info, self.system_utterance and self.current_state
        """
        self.pick_restaurant_option()

        if self.current_suggestion_info:
            self.system_utterance = self.utter('suggest_restaurant', fill=self.preferences,
                                               name=self.current_suggestion_info['restaurantname'])
            self.current_state = 's4_suggested_restaurant'

            # Add strings for how reasoning went
            if self.reasoning_preferences:
                if self.current_suggestion_info['satisfies_add_req'] == 'yes':
                    for field, val in self.reasoning_preferences.items():
                        field_str = field if val else 'not ' + field
                        chosen_template = random.choice(system_templates['reasoning']['direct'][field_str])
                        self.system_utterance += '\n\t' + chosen_template
                else:
                    chosen_template = random.choice(system_templates['reasoning']['indirect'])
                    reqs_str = ' and '.join([random.choice(system_templates['restaurant_info'][field][val])
                                             for field, val in self.reasoning_preferences.items()])
                    self.system_utterance += '\n\t' + chosen_template.format(reasoning_preferences = reqs_str)
        else:
            all_preferences = {}
            for key in self.preferences:
                all_preferences[key] = self.preferences[key]
            for key in self.reasoning_preferences:
                all_preferences[key] = self.reasoning_preferences[key]
            self.system_utterance = self.utter('no_restaurant_found',
                                               fill=all_preferences)
            self.current_state = 's8_no_restaurants_found'

    def handle_requests_confirms(self):
        """Extracts relevant fields as information about a suggested restaurant in case of either 'request' or 'confirm'
        speech acts. Calls back to utility in extract_preferences_requests.py.

        Returns:
            void (but fills self.system_utterance)
        """
        requests = extract_request_fields(self.user_input, self.dialogue_act == 'confirm')
        logging.debug(f'Extracted requests: {requests}')

        requests_data = {req: self.current_suggestion_info.get(req) for req in requests}
        self.system_utterance = (f"{self.utter('answer_request', fill=requests_data, request=True)}  "
                                 f"{random.choice(system_templates['additional_reqconf'])}")

    # STATE TRANSITIONING TEXT PROCESSORS ==============================================================================

    def process_commands(self) -> bool:
        """ Will process all the console commands entered by the user.

        Returns:
            A boolean that shows whether a single command was fired.
        """
        success = True

        match self.user_input:
            case r'\quit':
                print('System | Exiting... Goodbye!')
            case r'\help':
                print(f'\n{COMMANDS}')
            case r'\run_evaluation':
                self.evaluation_call(DF_DIALOGUE_ACTS, DF_DIALOGUE_ACTS_DD)
            case r'\restart_conversation':
                self.reset_conversation()
            case r'\change_system':
                self.active_system = None
                self.redetermine_system()
            case r'\toggle_w2v':
                self.w2v_on = not self.w2v_on
                print(f'System | W2V suggestions enabled: {self.w2v_on}')
            case r'\toggle_tts':
                if not self.tts_engine:
                    self.tts_engine = pyttsx3.init()
                self.tts_on = not self.tts_on
            case r'\stt_on':
                self.stt_on = not self.stt_on
            case r'\classification_testing_mode':
                self.classification_test_mode()
            case r'\toggle_debug':
                self.debug_on = not self.debug_on
                print(f'System | Debugging mode enabled: {self.debug_on}')
                if self.debug_on:
                    logging.getLogger().setLevel(logging.DEBUG)
                else:
                    logging.getLogger().setLevel(logging.WARNING)
            case str(x) if '\condition_' in x:
                if self.user_input[11] in 'Aa':
                    print('Starting conversation in condition A.')
                    self.condition = 'A'
                    self.reset_conversation()
                elif self.user_input[11] in 'Bb':
                    print('Starting conversation in condition B.')
                    self.condition = 'B'
                    self.reset_conversation()
                else:
                    print(f'Condition "{self.user_input[11]}" not recognised, please try again.')
            case _:
                success = False

        return success

    def process_baselines(self) -> bool:
        """ Will process all the baseline dialogue acts that hold in every state.

        Returns:
            A boolean that shows whether a single command was fired.
        """
        success = True

        match self.dialogue_act:
            case 'hello':
                self.system_utterance = self.utter('welcome') if not self.preferences else 'System | Hey there!'
            case 'restart':
                self.reset_conversation()
            case 'no_act':
                self.system_utterance = f"{self.utter('what?')}\n\t{self.system_utterance}"
            case 'repeat':
                pass
            case 'thankyou' | 'bye':
                self.end_conversation()
            case _:
                success = False

        return success

    def process_dialogue_states(self):
        """ This function contains the dialogue manager's main procedure. It generally contains two nested match-case
        statements: the outer one for the state that the dialogue is in, the inner one for the dialogue act that the
        user input was classified in.

        Returns:
            void, but fills self.system_utterance and self.current_state
        """

        match self.current_state:
            case 's1_welcomed':
                match self.dialogue_act:
                    case 'inform':
                        self.handle_first_prefs()
                    case _:
                        # No field specified in user input, ask for a random specific one
                        spec_field = random.choice(PREFERENCE_NAMES)
                        self.system_utterance = self.utter(f'request_{spec_field}')
                        self.current_state = f's6_requested_field_{spec_field}'
            case 's2_confirmed_selection':
                match self.dialogue_act:
                    case 'negate' | 'deny':
                        # Some preference is wrong, see if the negation message already contains new information
                        self.handle_prefs_update()
                    case _:
                        self.system_utterance = self.utter('ask_additional_reqs')
                        self.current_state = 's3_asked_additional_reqs'
            case 's3_asked_additional_reqs':
                self.reasoning_preferences = extract_add_requirements(self.user_input)
                if not self.reasoning_preferences:
                    # All preferences filled, let's suggest a restaurant
                    self.suggest_candidate()
                else:
                    if self.reasoning_preferences:
                        self.suggest_candidate()
                    else:  # System didn't find any requirements it can work with
                        self.system_utterance = f"{self.utter('what?')}\n\t{self.system_utterance}"
            case 's4_suggested_restaurant':
                match self.dialogue_act:
                    case 'reqalts' | 'negate' | 'deny' | 'reqmore':
                        # Suggest a different restaurant
                        self.suggest_candidate()
                    case 'request' | 'confirm':
                        self.handle_requests_confirms()
                        self.current_state = 's5_answered_request'
                    case 'bye' | 'thankyou':
                        self.end_conversation()
                    case _:  # Likely 'affirm' / 'ack'
                        self.system_utterance = self.utter('propose_request_or_goodbye')
                        self.current_state = 's9_proposed_request_or_goodbye'
            case 's5_answered_request':
                match self.dialogue_act:
                    case 'request' | 'confirm':
                        self.handle_requests_confirms()
                        self.current_state = 's5_answered_request'
                    case _:
                        self.system_utterance = self.utter('propose_request_or_goodbye')
                        self.current_state = 's9_proposed_request_or_goodbye'
            case str(x) if 's6_requested_field_' in x:
                requested_field = self.current_state.split('_')[-1]
                new_prefs = extract_preferences(self.user_input, self.w2v_on, spec_field=requested_field)
                self.preferences = update_preferences(self.preferences, new_prefs)

                field_content, success = new_prefs[requested_field]
                if not success:
                    if type(field_content) == str:
                        # This field had a Levenshtein edit distance > 3
                        self.system_utterance = self.utter('field_not_recognized').format(field=field_content)
                    elif type(field_content) == list:
                        # This field was not recognized but has an indirect mapping
                        self.suggest_indirect_mappings(requested_field, field_content)
                else:
                    self.check_if_prefs_filled()
            case 's7_queried_for_change':
                match self.dialogue_act:
                    case 'inform' | 'reqalts':
                        self.handle_prefs_update()
                    case _:
                        # Ask again
                        self.system_utterance = self.utter('what?')
            case 's8_no_restaurants_found':
                match self.dialogue_act:
                    case 'reqalts' | 'inform':
                        new_prefs = extract_preferences(self.user_input, self.w2v_on)
                        self.preferences = update_preferences(self.preferences, new_prefs)
                        self.system_utterance = self.utter('confirm_selection', fill=self.preferences)
                        self.current_state = 's2_confirmed_selection'
                    case _:
                        # Ask for a random field to change preferences
                        spec_field = random.choice(PREFERENCE_NAMES)
                        self.system_utterance = self.utter(f'request_{spec_field}')
                        self.current_state = f's6_requested_field_{spec_field}'
            case 's9_proposed_request_or_goodbye':
                match self.dialogue_act:
                    case 'request' | 'confirm':
                        self.handle_requests_confirms()
                        self.current_state = 's5_answered_request'
                    case 'bye' | 'thankyou':
                        self.end_conversation()
                    case _:
                        self.system_utterance = self.utter('what?')

    def text_to_speech(self):
        def sanitize_filename(name):
            name = name.replace(" ", "_").replace('\n', '_')
            return re.sub(r'[?/:*"<>|]', '_', name)

        def generate_file_name(utterance):
            m = hashlib.md5()
            m.update(utterance.encode('utf-8'))
            return m.hexdigest()

        VOICE_ID = '21m00Tcm4TlvDq8ikWAM'
        CHUNK_SIZE = 1024
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": "bb2599a986eff97f6f8da73db1a384c7"
        }

        data = {
            "text": self.system_utterance,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        # Later in your code...
        hashed_system_utterance = generate_file_name(self.system_utterance)
        save_path = os.path.join('code', 'recommendation_system', 'cached_audio', f"{VOICE_ID}_{hashed_system_utterance}.mp3")

        if not os.path.exists(save_path):
            response = requests.post(url, json=data, headers=headers)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

        mixer.music.load(save_path)
        mixer.music.play()


    # STATE TRANSITIONING MAIN =========================================================================================

    def transition_states(self):
        """ This function is called again for every user input. It prints the system output, takes user input and
        performs the associated actions. If it contains a command, it executes it. Otherwise, it uses the chosen
        classifier to determine the user's dialogue act. Based on the act, it may perform a baseline task or call the
        dialogue acts function.

        Returns:
            void, but fills self.user_input and self.dialogue_act
        """
        # Print (and optionally voice output) the selected system utterance relevant to the current state to the console
        logging.debug(f'Current state: {self.current_state}')
        print(f'\nSystem | {self.system_utterance}')

        if self.tts_on:
            self.text_to_speech()
            #self.tts_engine.say(self.system_utterance)
            #self.tts_engine.runAndWait()

        # Create line separation
        if self.debug_on:
            time.sleep(.25)
            sys.stdout.flush()

        # Get user input
        if self.stt_on:
            print()
            result = self.record_user_input()
            if result:
                time.sleep(.5)
                print(f'\n\nYou | {result}')
                self.user_input = result
            else:
                self.stt_on = not self.stt_on
                self.user_input = input('\nYou | ')
        else:
            self.user_input = input('You | ').lower()

        self.user_input = self.user_input.replace("’", "'")

        # Process console commands
        if self.user_input[0] == '\\':
            # If no commands are processed proceed with transition.
            if self.process_commands():
                return self.transition_states()

        # Classify user response
        act_classification = self.system_call(self.user_input, *self.model_deps)
        self.dialogue_act = act_classification[0] if isinstance(act_classification, tuple) else act_classification
        logging.info(f'User act: {self.dialogue_act}')
        logging.debug(f'User input classified as {self.dialogue_act}')

        # Baseline behaviour and act:action pairs which hold for every dialogue state
        # Reasoning not in training data so baseline has to be prevented.
        if not (self.current_state == 's3_asked_additional_reqs' and self.dialogue_act == 'no_act'):
            if self.process_baselines():
                return self.transition_states()

        # Handle the different dialogue states
        self.process_dialogue_states()

        return self.transition_states()


if __name__ == '__main__':
    dialogue = Dialogue(condition='A')
    dialogue.transition_states()
