import pandas as pd


def main():
    file_path = "data/glocare_pairs.csv"

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()