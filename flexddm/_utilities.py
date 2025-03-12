import pandas as pd 

def convertToDF(tuple_data, participant_id):
    """
    Converts simulated data from a tuple into a pandas DataFrame.
    """
    return pd.DataFrame({
        'id': [participant_id] * len(tuple_data[0]),
        'trial': tuple_data[0],
        'accuracy': tuple_data[1], 
        'rt': tuple_data[2],
        'condition': tuple_data[3]
        # condition_list is now an (nTrials x 3) array in your example –
        # store each column as condition_0, condition_1, condition_2, etc.
        # This means 'congruency' in the tuple is actually an array of shape (nTrials, #conditions).
        # We'll keep them in separate columns so you can group on them later.
        # 'condition_0': tuple_data[3][:,0],
        # 'condition_1': tuple_data[3][:,1],
        # 'condition_2': tuple_data[3][:,2],
    })


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