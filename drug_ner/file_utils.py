import os
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)


def get_gold_standard_file():
    return os.path.join(get_data_dir(), config['ground_truth'])


def get_gpt_prediction_eval_file():
    return os.path.join(get_data_dir(), config['gpt_prediction_file'])

def get_brief_summaries_eval_file():
    return os.path.join(get_data_dir(), config['brief_summaries_file'])

def get_t5_prediction_eval_file():
    return os.path.join(get_data_dir(), config['t5_prediction_file'])

def get_data_dir():
    """
         Get the absolute path to the 'data' directory located at the root of the project.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    return data_dir
