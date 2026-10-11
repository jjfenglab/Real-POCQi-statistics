"""
Prepare the dataset CSV for evaluation. The training-time pipeline leaves
`llm_output` empty for test rows (keyphrases are only extracted on train),
and `pandas.read_csv` interprets those blanks as NaN. Downstream
`load_data_partition` drops NaN rows, which empties the test partition.
Fill those NaN values with a placeholder so the test rows survive the filter.
"""
import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=str, required=True)
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--column", type=str, default="llm_output")
    p.add_argument("--fill", type=str, default="(no keyphrases extracted)")
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    n_filled = df[args.column].isna().sum()
    df[args.column] = df[args.column].fillna(args.fill)
    print(f"Filled {n_filled} NaN values in column {args.column!r}")
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
