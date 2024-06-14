import torch

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_metric
import numpy as np
import json
from tqdm import tqdm
import operator
import os
import ast
import ast

MODEL = 't5-small'
BATCH_SIZE = 8
NUM_PROCS = 16
EPOCHS = 4
OUT_DIR = 'results_t5small'
MAX_LENGTH = 128  # Maximum context length to consider while preparing dataset.

# Load database configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Get the path to the data directory
project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, 'data')

input_file = os.path.join(data_dir, config['gpt_prediction_file_training'])
df = pd.read_csv(input_file).replace(np.nan, '')


# Split the data into training and test sets
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
gd_drugs = df_test['drug_names'].head(50).tolist()


dataset_train = Dataset.from_pandas(df_train)
dataset_valid = Dataset.from_pandas(df_test)

tokenizer = T5Tokenizer.from_pretrained(MODEL)


# Function to convert text data into model inputs and targets
def preprocess_function(examples):
    """"
    Preprocesses the input examples for the model.
    """

    # define prompt
    inputs = [f"assign tag: {brief_summary}" for brief_summary in examples['brief_summary']]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    )

    # Set up the tokenizer for targets
    # define target text
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['drug_names'],
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




# Apply the function to the whole dataset
tokenized_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=1
)
tokenized_valid = dataset_valid.map(
    preprocess_function,
    batched=True,
    num_proc=1
)

# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Custom trainer class for T5 model
class T5Trainer(Trainer):

    def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix: str = "eval",
    ):

        # print('Evaluating')
        batch_size = 12
        all_clines = df_test['brief_summary'].tolist()
        gd_drugs = df_test['drug_names'].tolist()
        # ================================== EVALUATE ==================================


        i = 0
        pd_drugs = []
        with tqdm(total=len(all_clines)) as pbar:

            while i < len(all_clines):

                # ================================== BATCH INFERENCE==================================
                batch_texts = []
                batch_gd_target_texs = []
                while len(batch_texts) < batch_size and i < len(all_clines):
                    c_line = all_clines[i]
                    input_text = f"assign tag: {c_line}"
                    batch_texts.append(input_text)
                    try:
                        gd_text = gd_drugs[i].replace("\'", "\"")
                        gd_text = json.loads(gd_text)
                    except Exception:
                        gd_text = []
                    batch_gd_target_texs.append(gd_text)
                    i += 1
                    pbar.update(1)  # Update the progress bar
                # print(batch_texts, device)

                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)

                output_sequences = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_length=128,
                    num_beams=5,  # `num_beams=1` indicated temperature sampling.
                    early_stopping=True
                )

                batched_predictions = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                for pred in batched_predictions:
                    try:
                        pred = ast.literal_eval(pred)
                    except Exception:
                        pred = []
                    pd_drugs.append(pred)

        # compute P/R/F
        tp = 0
        fp = 0
        fn = 0
        for index, gold_row in enumerate(gd_drugs):
            gold_row = ast.literal_eval(gold_row)
            for gold_drug in gold_row:
                if gold_drug in pd_drugs[index]:
                    tp += 1
                else:
                    fn += 1

        for index, pred_row in enumerate(pd_drugs):
            for pred_drug in pred_row:
                gold_row = ast.literal_eval(gd_drugs[index])
                if pred_drug not in gold_row:
                    fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)

        print(precision, recall, fscore)


training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=OUT_DIR,
    logging_steps=500,
    evaluation_strategy='steps',
    save_steps=300,
    eval_steps=300,
    load_best_model_at_end=True,
    save_total_limit=5,
    learning_rate=0.001,
    fp16=True,
    dataloader_num_workers=0
)

trainer = T5Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid
)

history = trainer.train()
