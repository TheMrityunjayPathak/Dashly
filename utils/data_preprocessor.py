import pandas as pd
import numpy as np

def optimize_dataframe(df):
    """
    Optimizes a pandas DataFrame by:
    - Converting date columns to datetime format instead of object datatype
    - Downcasting integer and float columns to reduce memory usage
    - Converting object columns to category type if cardinality is low
    - Stripping leading/trailing whitespace from text columns

    Parameters:
        df (pd.DataFrame): The input DataFrame to be optimized.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    datetime_cols = df.filter(like="date").columns
    for column in datetime_cols:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    
    int_cols = df.select_dtypes(include="int64").columns
    for col in int_cols:
        df[col] = df[col].astype(np.int32)

    float_cols = df.select_dtypes(include="float64").columns
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

    categorical_cols = ["ship_mode", "segment", "country", "state", "region", "category", "sub_category"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    return df

def optimize_customers(df):
    """
    Optimizes customers DataFrame by:
    - Downcasting integer columns to reduce memory usage
    - Converting object columns to category type if cardinality is low
    - Stripping leading/trailing whitespace from text columns

    Parameters:
        df (pd.DataFrame): The input DataFrame to be optimized.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    int_cols = df.select_dtypes(include="int64").columns
    for col in int_cols:
        df[col] = df[col].astype(np.int32)

    categorical_cols = ["segment", "city", "state", "country", "region"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    return df

def optimize_orders(df):
    """
    Optimizes orders DataFrame by:
    - Converting date columns to datetime format instead of object datatype
    - Downcasting integer and float columns to reduce memory usage
    - Converting object columns to category type if cardinality is low
    - Stripping leading/trailing whitespace from text columns

    Parameters:
        df (pd.DataFrame): The input DataFrame to be optimized.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    datetime_cols = df.filter(like="date").columns
    for column in datetime_cols:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    int_cols = df.select_dtypes(include="int64").columns
    for col in int_cols:
        df[col] = df[col].astype(np.int32)

    float_cols = df.select_dtypes(include="float64").columns
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

    categorical_cols = ["ship_mode"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    return df