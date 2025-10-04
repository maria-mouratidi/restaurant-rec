# Restaurant Recommendation Dialogue System

## Overview

This project implements an intelligent restaurant recommendation dialogue system designed to study the impact of choice variety on user satisfaction and decision-making. The system serves as both a functional dialogue agent and an experimental platform for investigating human-computer interaction in decision-making scenarios.

### Research Question
*"How does manipulating choice variety influence user satisfaction and overwhelmedness in decision-making scenarios with a digital assistant?"*

The study explores whether reducing user agency by limiting choice options can lead to better satisfaction and reduced choice overwhelmedness, inspired by the "paradox of choice" theory. The system was designed to test two conditions:
- **Condition A (Control)**: Presents 2 cuisine sub-categories after user expresses preference
- **Condition B (Experimental)**: Presents 5 cuisine sub-categories for wider choice variety

### Key Features
- **Multiple Classification Models**: Rule-based, Decision Tree, Feed-Forward Neural Networks, and Majority Class baselines
- **Dialogue Act Classification**: Automatic classification of user intents (inform, request, confirm, etc.)
- **Restaurant Recommendation Engine**: Intelligent preference extraction and restaurant matching
- **Multimodal Interface**: Text-to-speech output and speech-to-text input capabilities
- **Comprehensive Evaluation**: Built-in performance metrics and confusion matrix generation
- **Experimental Framework**: A/B testing infrastructure for choice variety research

## Authors
- Teun Buwalda (t.c.buwalda@students.uu.nl)
- Maria Mouratidi (m.mouratidi@students.uu.nl)
- Aron Noordhoek (a.j.noordhoek@students.uu.nl)
- Alimohamed Jaffer (a.a.jaffer@students.uu.nl)  
- Jichen Li (j.li20@students.uu.nl)

## Research Methodology & Findings

### Experimental Design
- **Participants**: 42 participants (mean age: 27.26, 55% male, 36% female)
- **Design**: Within-subject experiment with three sub-tasks per participant
- **Conditions**: Random assignment to condition sequences (excluding 'AAA' and 'BBB')
- **Measurement**: Post-experimental Likert scale questionnaires measuring satisfaction and overwhelmedness

### Key Results
The study's findings challenged the initial hypothesis:
- **No significant difference** in satisfaction with recommendation process (p = 0.07)
- **No significant difference** in satisfaction with final restaurant choice (p = 0.52)  
- **No significant difference** in feeling of overwhelmedness (p = 0.22)
- **97% of permutations** showed small to medium positive effect of more choices on overall satisfaction
- **Effect sizes** were generally small, suggesting practical considerations beyond statistical significance

### Implications
The results suggest that the "paradox of choice" may not apply uniformly to all decision-making contexts, particularly in dialogue-based recommendation systems where other interaction factors may overshadow choice variety effects.

## Quick Start

```bash
# Clone and navigate to the project
cd restaurant-rec

# Set up Python environment (requires Python 3.10.11)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run the main dialogue system
python code/dialogue_system.py
```

### Available Commands in Dialogue System
- `\quit`: Exit application
- `\help`: Show all commands
- `\change_system`: Switch between classification models
- `\run_evaluation`: Run performance evaluation
- `\restart_conversation`: Restart with same system
- `\toggle_tts`: Enable/disable text-to-speech
- `\stt_on`: Enable speech-to-text input
- `\toggle_debug`: Show detailed logging

## File Structure
| File/Folder                                       | Description                                                                                         |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **`code/`**                                       | Main code directory                                                                                 |
| &emsp;**`decision_tree/`**                        | Decision tree Directory                                                                             |
| &emsp;&emsp;**`deduplicated_model/`**             | Deduplicated Decision tree model files                                                              |
| &emsp;&emsp;**`full_model/`**                     | Full model files                                                                                    |
| &emsp;&emsp;**`system_decision_tree.py`**         | Decision tree system                                                                                |
| &emsp;&emsp;**`train_decision_tree.py`**          | Decision tree training                                                                              |
| &emsp;**`feed_forward/`**                         | Feed forward neural net directory                                                                   |
| &emsp;&emsp;**`deduplicated_model/`**             | Deduplicated model files for feed forward                                                           |
| &emsp;&emsp;**`full_model/`**                     | Full model files for feed forward                                                                   |
| &emsp;&emsp;**`system_feed_forward.py`**          | Feed forward system                                                                                 |
| &emsp;&emsp;**`train_feed_forward.py`**           | Feed forward training                                                                               |
| &emsp;**`majority_class/`**                       | Majority class model directory                                                                      |
| &emsp;&emsp;**`exploration_majority_class.py`**   | Data exploration for majority class system                                                          |
| &emsp;&emsp;**`system_majority_class.py`**        | Majority class system                                                                               |
| &emsp;**`rule_based/`**                           | Rule-based model directory                                                                          |
| &emsp;&emsp;**`dialogue_act_rules.yml`**          | Dialogue rules for each act in yaml format.                                                         |
| &emsp;&emsp;**`exploration_rule_based.py`**       | Data exploration for development of Rule-based System.                                              |
| &emsp;&emsp;**`system_rule_based.py`**            | Rule based system                                                                                   |
| &emsp;**`recommendation_system/`**                | Recommendation system folder (utils for main Dialogue app)                                          |
| &emsp;&emsp;**`extract_preferences_requests.py`** | Main script for the extraction of both preferences and requests/confirms.                           |
| &emsp;&emsp;**`field_mappings.py`**               | Preference field mappings, both indirect and direct.                                                |
| &emsp;&emsp;**`system_templates.py`**             | System Templates, multiple system utterences per action to avoid robotic conversation.              |
| &emsp;&emsp;**`word2vec files`**                  | The stored vectors, available upon first run.                                                       |
| &emsp;**`constants.py`**                          | Constants file to avoid redundancy and mutability.                                                  |
| &emsp;**`data_processing.py`**                    | Data Processing file for reasoning and .dat file.                                                   |
| &emsp;**`dialogue_system.py`**                    | !!! MAIN CONSOLE APP !!! contains the Dialogue class and initializes the conversation upon running. |
| &emsp;**`model_evaluation.py`**                   | Main model evaluation file used by every system inside the console app.                             |
| **`data/`**                                       | Data directory                                                                                      |
| &emsp;**`co_occurrences/`**                       | Co-occurrences directory based on training data for each act.                                       |
| &emsp;**`all_dialogs.txt`**                       | Dialogs file                                                                                        |
| &emsp;**`dialog_acts.csv`**                       | Dialogue Acts .csv version of supplied .dat file                                                    |
| &emsp;**`dialog_acts.dat`**                       | Original .dat acts file                                                                             |
| &emsp;**`restaurant_info.csv`**                   | Restaurant Info, enriched for reasoning                                                             |
| **`evaluation/`**                                 | Evaluation directory in which metrics and confusion matrices are saved for each system.             |
| **`assets/`**                                     | Assets such as distribution plots and wordclouds                                                    |
| &emsp;**`wordclouds/`**                           | Wordclouds based on training data for development of Rule system                                    |
| **`MAIR-RRDS_Dialogue_Model.png`**                | Our Dialogue Model in photo format                                                                  |
| **`README.md`**                                   | Current File                                                                                        |
| **`requirements.txt`**                            | Requirements from pip freeze, see installation info.                                                |


## Classification Models & Architecture

### Dialogue Act Classification Systems

The system implements multiple approaches for classifying user utterances into dialogue acts:

#### 1. **Rule-Based System** (`rule_based/`)
- Uses hand-crafted linguistic rules defined in YAML format
- Pattern matching for keywords and phrases
- Baseline approach with high interpretability
- Rules stored in `dialogue_act_rules.yml`

#### 2. **Decision Tree** (`decision_tree/`)
- Scikit-learn implementation with bag-of-words features
- Available in both full dataset and deduplicated versions
- Provides interpretable decision paths
- Stored models in `full_model/` and `deduplicated_model/` directories

#### 3. **Feed-Forward Neural Network** (`feed_forward/`)
- TensorFlow/Keras implementation
- Uses Word2Vec embeddings (300-dimensional)
- Multiple hidden layers with dropout for regularization
- Separate models for full and deduplicated datasets
- **Note**: Models need to be trained first due to size constraints

#### 4. **Majority Class Baseline** (`majority_class/`)
- Always predicts the most frequent dialogue act ('inform')
- Used as baseline for performance comparison

### Dialogue Acts Supported
- **inform**: User provides information about preferences
- **request**: User asks for specific information
- **confirm**: User confirms system suggestions
- **deny/negate**: User rejects suggestions
- **affirm**: User agrees with system
- **hello/bye**: Greeting and farewell
- **thankyou**: Gratitude expressions
- **reqalts**: Request alternatives
- **reqmore**: Request more information
- **repeat**: Ask for repetition
- **restart**: Start conversation over

### Recommendation Engine (`recommendation_system/`)
- **Preference Extraction**: Uses Word2Vec similarity for flexible matching
- **Field Mappings**: Maps user utterances to restaurant attributes (cuisine, price, area)
- **Template System**: Varied response templates to avoid repetitive interactions
- **Restaurant Database**: Enhanced CSV with 33 restaurants across different categories

## Datasets & Evaluation

### Data Sources
- **Dialog Acts**: 15,611 labeled utterances for dialogue act classification
- **Restaurant Database**: 33 restaurants with attributes (name, cuisine, price, area, phone, address)
- **Experimental Data**: Results from 42 participants across 126 dialogue sessions

### Performance Metrics
- **Accuracy**: Primary metric for dialogue act classification
- **Confusion Matrices**: Detailed error analysis per dialogue act
- **User Satisfaction**: Likert scale ratings (1-5) for recommendation process and final choice
- **Overwhelmedness**: Self-reported feelings of choice anxiety

### Evaluation Results (Classification Performance)
Generated confusion matrices and performance reports available in `evaluation/` directory for each model:
- Decision Tree: Typically achieves ~85-90% accuracy
- Feed-Forward NN: ~80-85% accuracy with better generalization
- Rule-Based: ~70-75% accuracy but highly interpretable
- Majority Class: ~14% accuracy (baseline)

## Data Visualization & Analysis

The project includes comprehensive visualizations:
- **Word Clouds** (`assets/wordclouds/`): Most common words per dialogue act
- **Distribution Plots**: Dialogue act frequency analysis
- **Training Curves**: Loss and validation curves for neural networks
- **Confusion Matrices**: Model performance analysis
- **Demographic Plots**: Participant characteristics and experimental results

## Technical Notes & Troubleshooting

### Important Setup Notes
- **Feed-Forward Model**: Due to size constraints, the trained model is not included in the repository. Run `python code/feed_forward/train_feed_forward.py` to generate it locally.
- **First Run Downloads**: Word2Vec vectors and embedding models (~1.5GB) will be downloaded automatically on first execution.
- **Natural Language Processing**: The system uses Word2Vec embeddings for more natural language understanding. Simple keyword matching (like other groups might use) is less effective but more realistic for practical applications.
- **Path Management**: All scripts automatically adjust to the project root directory. If path issues occur, run scripts from the base `mair-rrds` folder.

### Performance Characteristics
- **Embedding-based Classification**: More natural but less overfitted than bag-of-words approaches
- **Generalization**: Better handling of variations like "yeah" vs "yes" for affirm classification
- **Natural Interaction**: Designed for conversational sentences rather than keyword shortcuts

## Requirements and detailed usage
python 3.10.11 (https://www.python.org/downloads/release/python-31011/)

## Usage Examples

### Basic Conversation Flow
```
System: Hello, welcome to the restaurant recommendation system! How can I help you?
User: I want a cheap restaurant
System: What kind of food would you like?
User: Asian food
System: We have chinese, korean. What would you like?
User: Chinese please
System: The Rice House is a nice chinese restaurant in the cheap price range.
```

### Model Switching Example
```
User: \change_system
System: Available systems: majority_class, rule_based, decision_tree-full, decision_tree-dd, feed_forward-full, feed_forward-dd
System: Enter system name:
User: decision_tree-full
System: Switched to decision_tree-full
```

### Training a New Model
```bash
# Train decision tree model
python code/decision_tree/train_decision_tree.py

# Train feed-forward neural network
python code/feed_forward/train_feed_forward.py

# Evaluate rule-based system
python code/rule_based/exploration_rule_based.py
```

### Running Model Evaluation
```bash
# From within the dialogue system
User: \run_evaluation
# Or directly
python code/model_evaluation.py
```

### Debug: static type checking
```bash
mypy --ignore-missing-imports code/dialogue_system.py
```

## Example Dialogue Acts Classification

| User Input | Predicted Act | Confidence |
|------------|---------------|------------|
| "I want Italian food" | inform | 0.95 |
| "Can you give me the phone number?" | request | 0.88 |
| "Yes, that sounds good" | affirm | 0.92 |
| "No, I don't like that" | deny | 0.87 |
| "Do you have anything else?" | reqalts | 0.84 |

## Project Structure for Development

### Adding a New Classification Model
1. Create directory in `code/` (e.g., `svm/`)
2. Implement `system_[model].py` with `classify_[model]()` function
3. Add training script `train_[model].py`
4. Update `constants.py` SYSTEMS list
5. Import functions in `dialogue_system.py`

### Modifying Restaurant Database
- Edit `data/restaurant_info.csv`
- Update `recommendation_system/field_mappings.py` for new attributes
- Regenerate any cached embeddings if needed

## Research Impact & Applications

This project contributes to several areas of AI and HCI research:

### Academic Contributions
- **Choice Architecture in AI Systems**: Empirical evidence on choice variety effects in dialogue systems
- **Dialogue Act Classification**: Comparative analysis of multiple ML approaches on restaurant domain
- **Human-AI Collaboration**: Insights on agency allocation between users and systems

### Practical Applications
- **Conversational AI Development**: Framework for building domain-specific dialogue systems
- **A/B Testing Infrastructure**: Methodology for testing different interaction designs
- **Multi-modal Interfaces**: Integration of text, speech, and audio feedback
