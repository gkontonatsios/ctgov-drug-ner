import pandas as pd
from file_utils import get_gold_standard_file, get_gpt_prediction_eval_file, get_t5_prediction_eval_file


gold_df = pd.read_csv(get_gold_standard_file())
# pred_df = pd.read_csv(get_gpt_prediction_eval_file())
pred_df = pd.read_csv(get_t5_prediction_eval_file())

# Extract the preferred drug names
gold_drugs = gold_df['preferred_drug_names'].apply(eval).tolist()
pred_drugs = pred_df['preferred_drug_names'].apply(eval).tolist()

tp = 0
fp = 0
fn = 0
# Calculate true positives and false negatives

for index, gold_row in enumerate(gold_drugs):
    pred_row = pred_drugs[index]
    for gold_drug in gold_row:
        if gold_drug in pred_row:
            tp += 1
        else:
            fn += 1

# Calculate false positives
for index, pred_row in enumerate(pred_drugs):
    gold_row = gold_drugs[index]
    for pred_drug in pred_row:
        if pred_drug.strip() == '':
            continue
        if pred_drug not in gold_row:
            fp += 1


# Calculate precision, recall, and F1 score
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (fn + tp) if (fn + tp) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(tp, fp, fn)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')