#!/usr/bin/env bash
set -euo pipefail

# Usage: run_analysis.sh [data_dir] [mapping_path]
data_dir="${1:-pilot_data}"
mapping_path="${data_dir}/secret_mapping.csv"

echo "## Extracting answer lengths from JSONL..."
python3 extract_answer_lengths.py "${data_dir}"

echo "## Blinding vendor identities..."
Rscript blind_vendors.R "${data_dir}" "${mapping_path}"

echo "## Knitting descriptives.Rmd..."
Rscript -e "rmarkdown::render('descriptives.Rmd', params=list(data_dir='${data_dir}'), output_dir='${data_dir}')"

echo "## Knitting analysis.Rmd (bootstrap by question)..."
Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}', bootstrap_by='question'), output_file='analysis_by_question.html', output_dir='${data_dir}')"

echo "## Knitting analysis.Rmd (bootstrap by user)..."
Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}', bootstrap_by='user'), output_file='analysis_by_user.html', output_dir='${data_dir}')"

## Example code for running analysis where you drop only a single vendor from the analysis
# echo "## Knitting analysis.Rmd (bootstrap by question, drop vendor A)..."
# Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}', bootstrap_by='question', drop_vendor='A'), output_file='analysis_by_question_drop_A.html', output_dir='${data_dir}')"
