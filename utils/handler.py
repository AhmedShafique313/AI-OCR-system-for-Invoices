import pandas as pd

def save_to_csv(json_data, file_path):
    """Converts JSON data to CSV format."""
    df = pd.DataFrame([json_data])
    df.to_csv(file_path, index=False)
