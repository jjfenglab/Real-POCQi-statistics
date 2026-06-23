"""Plot distribution of question types from tagged questions."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot question type distribution")
    parser.add_argument("--input-csv", required=True, help="Tagged questions CSV")
    parser.add_argument("--output-plot", required=True, help="Output plot path")
    return parser.parse_args()


def format_label(label: str) -> str:
    """Convert TAG_NAME to Title Case."""
    return label.replace("_", " ").title()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    counts = df["primary_question_type"].value_counts().sort_values()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("Blues_d", n_colors=len(counts))
    sns.barplot(x=counts.values, y=[format_label(l) for l in counts.index],
                palette=colors, ax=ax)

    ax.set_xlabel("Number of Questions", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Distribution of Primary Question Types", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
