from pathlib import Path

import pandas as pd


INPUT_FILE = Path("results/golden_review_sheet.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    print("Golden review sheet columns:")
    print("-" * 60)

    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    print("\nTotal columns:", len(df.columns))
    print("Total rows:", len(df))


if __name__ == "__main__":
    main()