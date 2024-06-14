import json
from collections import namedtuple

import psycopg2
from psycopg2 import sql
from psycopg2 import OperationalError
from loguru import logger
import pandas as pd
import numpy as np

# Load database configuration from JSON file
with open('config.json', 'r') as f:
    db_params = json.load(f)

def get_psql_connection():
    """
    Establishes a connection to a PostgreSQL database.

    :param dbname: Name of the database
    :param user: Username to connect to the database
    :param password: Password for the user
    :param host: Database host address (default is 'localhost')
    :param port: Port number (default is 5432)
    :return: Connection object or None if connection fails
    """
    try:
        connection = psycopg2.connect(
            dbname=db_params['dbname'],
            user=db_params['user'],
            password=db_params['password'],
            host=db_params['host'],
            port=db_params['port']
        )
        logger.info("Connection to PostgreSQL DB successful")
        return connection
    except OperationalError as e:
        logger.error(f"The error '{e}' occurred")
        return None


def load_drugcentral_synonyms():
    connection = get_psql_connection()
    cursor = connection.cursor()

    # SQL query to fetch data from the table
    query = """
    SELECT syn_id, id, name, preferred_name, parent_id, lname
    FROM public.synonyms
    """

    df = pd.read_sql_query(query, connection).replace(np.nan, '')
    logger.info(f"Succesfully loaded {df.shape[0]} records from public.synonyms")

    return df


def main():
    load_drugcentral_synonyms()


if __name__ == '__main__':
    main()
