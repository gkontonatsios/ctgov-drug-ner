from typing import Dict

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import pandas as pd
import json

from drug_ner.drugs import DrugsDB
import ast
import file_utils
from tqdm import tqdm

# initialise drugs_db
drug_db = DrugsDB()


class T5Inference:
    def __init__(self, checkpoint, model_name='t5-small', device='cpu'):
        print('Initialising T5 model')
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint)
        self.model.to(device)
        print('Done')

    def tag_sentence(self, sentence):
        input_text = f"assign tag: {sentence}"
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=128,
            padding='max_length',
            truncation=True
        ).to(self.device)

        # model.to(device)
        # Get correct sentence ids.
        corrected_ids = self.model.generate(
            inputs,
            max_length=128,
            num_beams=5,  # `num_beams=1` indicated temperature sampling.
            early_stopping=True
        )

        # Decode.
        predicted_entities = self.tokenizer.decode(
            corrected_ids[0],
            skip_special_tokens=True
        )
        predicted_entities = predicted_entities.replace("\'", "\"")
        predicted_entities = ast.literal_eval(predicted_entities)
        return predicted_entities

    def get_drug_names(self, input_summary: str) -> Dict[str, str]:
        # map drug names to preferred names
        drug_names = self.tag_sentence(input_summary)
        drug_names_with_mapped_preferred_names = {
            drug_name: drug_db.get_preferred_name(input_name=drug_name)
            for drug_name in drug_names
        }

        return drug_names_with_mapped_preferred_names


if __name__ == '__main__':
    from timeit import default_timer as timer

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    start = timer()

    t5_inference = T5Inference(
        checkpoint=config['t5_check_point'])

    # sentence = 'Multicenter, randomized, double-blind, placebo-controlled, 5-arm, dose-ranging study to assess the efficacy of subcutaneous injections of Golimumab (CNTO 148), 50 or 100 mg, at either 2- or 4- week intervals in subjects with active RA despite MTX therapy.'
    # sentence = 'The purpose of this study is to evaluate the effectiveness and safety of CNTO 148 (golimumab) in patients with severe persistent asthma.'
    # sentence = 'The purpose of the study is to determine whether an investigational compound, ALX-0600, is safe and effective in treating Crohn\'s Disease.'
    summary = 'The purpose of this study is to evaluate the effectiveness and safety of CNTO 1275 (ustekinumab) in patients with psoriatic arthritis.'
    drug_names_with_mapped_preferred_names = t5_inference.get_drug_names(input_summary=summary)

    nct_summaries_df = pd.read_csv(file_utils.get_brief_summaries_eval_file())

    drug_names = []
    pref_names = []
    for _, row in tqdm(nct_summaries_df.iterrows(), total=nct_summaries_df.shape[0], desc='Tagging drugs using T5'):
        local_names = t5_inference.get_drug_names(input_summary=row['brief_summary'])
        drug_names.append(list(local_names.keys()))
        pref_names.append(list(local_names.values()))

    nct_summaries_df["drug_names"] = drug_names
    nct_summaries_df["preferred_drug_names"] = pref_names
    nct_summaries_df.to_csv(file_utils.get_t5_prediction_eval_file(), index=False)
