"""Plot distribution of question types from tagged questions."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot question type distribution")
    parser.add_argument("--input-csv", required=True, help="Tagged questions CSV")
    parser.add_argument("--output-plot", required=True, help="Output plot path")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    counts = df["primary_question_type"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("Question Type")
    ax.set_title("Distribution of Primary Question Types")
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
