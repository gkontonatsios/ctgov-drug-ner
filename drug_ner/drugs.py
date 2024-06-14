import pandas as pd
from sqlutils import load_drugcentral_synonyms
from loguru import logger


class DrugsDB:
    def __init__(self):
        """
        Initializes the DataFrameHolder with a DataFrame.

        :param data: The data to initialize the DataFrame.
        """
        self.df = pd.DataFrame()

        self.load_db()

    #
    def load_db(self):
        """Load drug central from postrgres"""
        self.df = load_drugcentral_synonyms()

    def print_db(self):
        print(self.df.head(20).to_string())

    def get_preferred_name(self, input_name):
        """
            Function to get the preferred name for a given name
        """

        # logger.info(f"Getting preferred name for: {input_name}")

        # Convert input_name to lowercase
        input_name_lower = input_name.lower()

        # Find the row where 'name' matches the input name
        input_row = self.df[self.df['lname'] == input_name_lower]

        if not input_row.empty:
            # Get the id for the input name
            input_id = input_row.iloc[0]['id']

            # Filter the dataframe to get all names with the same id
            group_df = self.df[self.df['id'] == input_id]

            # Find the preferred name within the group
            preferred_row = group_df[group_df['preferred_name'] != '']

            if not preferred_row.empty:
                # logger.info(f'Preferred name found for {input_name}')
                return preferred_row.iloc[0]['name']

        return ""


def main():
    drugs = DrugsDB()
    print(drugs.get_preferred_name(input_name='IMMU-132'))


if __name__ == '__main__':
    main()
