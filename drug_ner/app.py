import streamlit as st
from annotated_text import annotated_text

from ctgov_crawler import get_brief_summary_by_nct_id
from openai_ner import get_drug_names
import re
from t5_inference import T5Inference
import json


def get_highlighted_text(summary, predicted_drug_names):
    """
        Highlights drug names in the clinical trial summary.

        Args:
            summary (str): The brief summary of the clinical trial.
            predicted_drug_names (dict): A dictionary with drug names and their preferred names.

        Returns:
            list: A list of tuples containing the tokens and their annotations.
        """
    summary = summary.replace(',', ' , ').replace('®', ' ® ').replace('(', ' ( ').replace(')', ' ) ').replace('/', ' / ')

    # Find and store the positions of the drug names in the summary
    annotated_spans = dict()
    for drug_name, preferred_name in predicted_drug_names.items():
        if preferred_name == '':
            continue
        matches = [m.start() for m in re.finditer(re.escape(drug_name.lower()), summary.lower())]
        for m in matches:
            if drug_name in annotated_spans:
                annotated_spans[drug_name].append((m, m + len(drug_name)))
            else:
                annotated_spans[drug_name] = [(m, m + len(drug_name))]

    annotated_tokens = []

    word_index = 0
    for word in summary.split(' '):
        if word.strip() == '':
            continue
        word_begin = summary.index(word, word_index)
        word_end = word_begin + len(word)
        word_index = word_end

        preferred_name = ''
        is_end_tag = False

        for drug_name, matched_spans in annotated_spans.items():

            for span in matched_spans:
                if word_begin >= span[0] and word_end <= span[1]:
                    # print(word, word_begin, word_end, span, word_begin >= span[0], word_end <= span[1])
                    preferred_name = predicted_drug_names[drug_name]
                    if str(drug_name.lower()).endswith(word.lower()):
                        is_end_tag = True
                    break
        if preferred_name != '' and not is_end_tag:
            annotated_tokens.append((word + ' ', '', 'orange'))
        elif preferred_name != '' and is_end_tag:
            annotated_tokens.append((word + ' ', preferred_name, 'orange'))
        else:
            annotated_tokens.append((word + ' '))
    return annotated_tokens


# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize the T5 inference model
t5_inference = T5Inference(checkpoint=config['t5_check_point'])

# Title of the app
st.title("Clinical trial summary")

# Sidebar for input
st.sidebar.header("Input")

# Input field for NCT ID
# NCT00175877
# nct_id = st.sidebar.text_input("Enter NCT ID:", 'NCT00074438')
nct_id = st.sidebar.text_input("Enter NCT ID:", 'NCT00175877')
# Dropdown menu for model selection
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ("GPT-3.5", "T5 (fine-tuned)")
)

# Run button
run_button = st.sidebar.button("Run")

# Display the NCT ID when the button is pressed
if run_button:
    with st.spinner('Fetching brief summary from CTGOV'):
        brief_summary = get_brief_summary_by_nct_id(nct_id=nct_id)

    predicted_drug_names = {}

    with st.spinner(f'Extracting drug names using: {model_choice}'):
        if model_choice == 'GPT-3.5':
            predicted_drug_names = get_drug_names(input_summary=brief_summary)
        else:
            predicted_drug_names = t5_inference.get_drug_names(input_summary=brief_summary)

    annotated_tokens = get_highlighted_text(brief_summary, predicted_drug_names)
    annotated_text(annotated_tokens)
