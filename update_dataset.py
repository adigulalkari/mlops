import pandas as pd
import argparse

def import_data(file_path):
    data = pd.read_csv(file_path)
    # Delteing randomly few lines from the data
    data = data.sample(frac=0.8, random_state=1).reset_index(drop=True)
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Update the dataset by removing some rows.")
    parser.add_argument("--dataset_path", required=True, help="Path to the CSV file containing the dataset.")
    args = parser.parse_args()

    data = import_data(args.dataset_path)
    data.to_csv(args.dataset_path, index=False)
    print(f"Updated dataset saved to {args.dataset_path}")