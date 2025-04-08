import pandas as pd 
import numpy as np

def convertToDF(tuple_data, participant_id):
    """
    Converts simulated data from a tuple into a pandas DataFrame.
    Combines multi-column condition array into a single string per row: 'x-y-z'.
    """
    condition_array = np.array(tuple_data[3])  # shape: (n_trials, 3)

    # Build the DataFrame
    sim_data = pd.DataFrame({
        'id': [participant_id] * len(tuple_data[0]),
        'trial': tuple_data[0],
        'accuracy': tuple_data[1], 
        'rt': tuple_data[2],
    })

    # Combine condition columns into a single string column like "1-0-1"
    sim_data['condition'] = [
        f"{row[0]}-{row[1]}-{row[2]}" for row in condition_array
    ]

    return sim_data


def getRTData(path, input_data_id, input_data_condition, input_data_rt, input_data_accuracy):
    """
    Loads and processes reaction-time data with multiple conditions dynamically.

    @path: str - Path to the data file.
    @input_data_id: str - Column name for participant ID.
    @input_data_condition: str - Column name that may contain single or multiple conditions
                                 (e.g. “1-0-0” vs. “0-0-1”).
    @input_data_rt: str - Column name for reaction time.
    @input_data_accuracy: str - Column name for accuracy.

    Returns:
    A pandas DataFrame with columns:
        id, condition (raw string),
        condition_0, condition_1, … (splits if dashes are found),
        rt, accuracy
    """
    data = pd.read_csv(path)

    # Extract the columns we need
    data = pd.DataFrame({
        'id':        data[input_data_id], 
        'condition': data[input_data_condition].astype(str),
        'rt':        data[input_data_rt], 
        'accuracy':  data[input_data_accuracy]
    })

    # If the 'condition' column has dashes, split into multiple columns: condition_0, condition_1, etc.
    if data['condition'].str.contains('-').any():
        condition_split = data['condition'].str.split('-', expand=True).astype(int)
        # Give them meaningful column names
        condition_split.columns = [f"condition_{i}" for i in range(condition_split.shape[1])]
        # Merge into data
        data = pd.concat([data, condition_split], axis=1)

    return data