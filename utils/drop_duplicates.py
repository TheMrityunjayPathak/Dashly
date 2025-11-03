def remove_duplicates(df):
    """
    Checks for duplicate rows in a pandas DataFrame and removes them if found.
    - If no duplicates are found, do nothing.
    - If duplicates are found, removes them.

    Parameters:
        df (pd.DataFrame): The DataFrame to check for and remove duplicate rows.

    Returns:
        pd.DataFrame: A DataFrame with duplicates removed (if any).

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 2], 'B': [3, 4, 4]})
        >>> df_cleaned = remove_duplicates(df)
        >>> df_cleaned
           A  B
        0  1  3
        1  2  4
    """
    if df.duplicated().sum() == 0:
        pass
    else:
        df.drop_duplicates(inplace=True)
    
    return df