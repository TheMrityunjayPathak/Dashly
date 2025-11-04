import pandas as pd

def append_unique_data_to_db(table_name, new_data, primary_key_column, engine):
    """
    Function to append new and unique data to the given table in the database.
    
    Parameters:
        table_name (str): Name of the target table in the database
        new_data (pd.DataFrame): DataFrame containing the new data to be appended
        primary_key_column (str): The column that serves as the primary key (e.g., 'customer_id', 'order_id')
        engine (SQLAlchemy Engine): SQLAlchemy engine to connect to the database

    Returns:
        None
    """
    existing_data = pd.read_sql(f"SELECT {primary_key_column} FROM {table_name}", engine)
    existing_ids = set(existing_data[primary_key_column])

    new_data_unique = new_data[~new_data[primary_key_column].isin(existing_ids)]

    if not new_data_unique.empty:
        new_data_unique.to_sql(table_name, engine, if_exists="append", index=False)