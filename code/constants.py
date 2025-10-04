import yaml
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)
sys.path.append('code')
from model_evaluation import prepare_data

MAJORITY_CLASS = 'inform'
RANDOM_SEED = 42
SYSTEMS = [
    'majority_class',
    'rule_based',
    'decision_tree-full',
    'decision_tree-dd',
    'feed_forward-full',
    'feed_forward-dd',
]
COMMANDS = """Commands:
    \\quit: exit application
    \\help: re-print commands
    \\change_system: switch to a different system
    \\run_evaluation: run performance evaluation for current system (full & deduplicated data) 
    \\restart_conversation: restart the conversation with the same system
    \\toggle_w2v: (default: on) turn on or off additional preference suggestions based on Word2Vec
    \\toggle_tts: (default: off) turn text-to-speech output on and off
    \\stt_on: (default: off) turn speech-to-text input on, press 'Esc' to turn off.
    \\classification_testing_mode: enter a mode where you can get act classifications for test utterances.
    \\toggle_debug: (default: off) enter a mode where the inner workings are shown through logging messages
"""
INFERENCE_RULES = [
    ('touristic', lambda x: (x['pricerange'] == 'cheap') & ((x['quality'] == 'good') | (x['quality'] == 'excellent')), True),
    ('touristic', lambda x: x['food'] == 'romanian', False),
    ('assignedseats', lambda x: x['crowdedness'] == 'busy', True),
    ('children', lambda x: x['lengthofstay'] == 'long', False),
    ('romantic', lambda x: x['crowdedness'] == 'busy', False),
    ('romantic', lambda x: x['lengthofstay'] == 'long', True)
]
with open(os.path.join('code', 'rule_based', 'dialogue_act_rules.yml'), 'r') as f:
    RULES = yaml.safe_load(f)
DF_DIALOGUE_ACTS, DF_DIALOGUE_ACTS_DD = prepare_data()
PREFERENCE_NAMES = ['food', 'pricerange', 'area']
COS_SIM_THRESHOLD = .5
