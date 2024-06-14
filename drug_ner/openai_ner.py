import pandas as pd
from typing import List, Dict
import json
from openai import OpenAI
import os
from drugs import DrugsDB
from tqdm import tqdm

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

client = OpenAI(api_key=config['open_ai_api_key'])

# initialise drugs_db
drug_db = DrugsDB()


def get_drug_names(input_summary: str) -> Dict[str, str]:
    """
        Extracts drug names from a clinical trial summary using OpenAI's GPT-3.5-turbo model.

        This method sends a prompt to the GPT-3.5-turbo model, asking it to extract all drug names from the provided clinical trial summary.
        The prompt includes examples of how drug names are expected to be identified within the text.

        Args:
            input_summary (str): The brief summary from which to extract drug names.

        Returns:
            List[str]: A list of drug names extracted from the summary.

        Example:
            input_summary = "This study evaluates the effectiveness of ibuprofen and acetaminophen in treating headaches. Participants were given either a placebo or Lipitor to assess cholesterol levels."
            drug_names = get_drug_names(input_summary)
            # Output: ['ibuprofen', 'acetaminophen', 'Lipitor']
        """
    PROMPT_TEMPLATE = (
        "Extract all drug names from the following clinical trial summary. "
        "The drug names may be brand names or generic drug names. You should avoid names that do not refer to a specific drug, such as placebo. Here are some examples:\n\n"
        "Example 1: This study evaluates the effectiveness of ibuprofen and acetaminophen in treating headaches. "
        "Drug names: ibuprofen, acetaminophen.\n\n"
        "Example 2: Participants were given either a placebo or Lipitor to assess cholesterol levels. "
        "Drug names: Lipitor.\n\n"
        "Now, extract drug names from this summary:\n\n"
        f"{input_summary}\n\nDrug names:"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Model = should match the deployment name you chose for your 0125-Preview model deployment
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful data analyst processing summaries from clinicaltrials.gov. Your output should be in json. As key of the json response use 'drug_names'"},
            {"role": "user", "content": PROMPT_TEMPLATE}
        ]
    )

    # map drug names to preferred names
    drug_names = json.loads(response.choices[0].message.content)['drug_names']
    drug_names_with_mapped_preferred_names = {
        drug_name: drug_db.get_preferred_name(input_name=drug_name)
        for drug_name in drug_names
    }

    return drug_names_with_mapped_preferred_names


def main():
    # summary_text = 'This is a multicenter, Phase 3 randomized, placebo-controlled study designed to evaluate adalimumab in children 4 to 17 years old with polyarticular juvenile idiopathic arthritis (JIA) who are either methotrexate (MTX) treated or non-MTX treated.'
    # drug_names = get_drug_names(input_summary=summary_text)



    # Get the path to the data directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    nct_summaries_df = pd.read_csv(os.path.join(data_dir, config['brief_summaries_training_file']))

    drug_names = []
    pref_names = []
    for _, row in tqdm(nct_summaries_df.iterrows(), total=nct_summaries_df.shape[0], desc='Tagging drugs using OpenAI GPT'):
        local_names = get_drug_names(input_summary=row['brief_summary'])
        drug_names.append(list(local_names.keys()))
        pref_names.append(list(local_names.values()))

    nct_summaries_df["drug_names"] = drug_names
    nct_summaries_df["preferred_drug_names"] = pref_names

    nct_summaries_df.to_csv(os.path.join(data_dir, config['gpt_prediction_file_training']), index=False)
if __name__ == '__main__':
    main()
