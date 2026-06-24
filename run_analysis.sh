#!/usr/bin/env bash
run() {
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

# Usage: run_analysis.sh [data_dir] [mapping_path]
data_dir="${1:-pilot_data}"
mapping_path="${data_dir}/secret_mapping.csv"
output_dir="${data_dir}/output"

run mkdir -p "${output_dir}"

run python3 extract_answer_lengths.py "${data_dir}"
run Rscript blind_vendors.R "${data_dir}" "${mapping_path}"

echo "## Knitting descriptives.Rmd..."
echo "   -> Outputs: descriptives.html, .RData"
run Rscript -e "rmarkdown::render('descriptives.Rmd', params=list(data_dir='${data_dir}', output_dir='${output_dir}', output_name='descriptives'), output_dir='${output_dir}')"

echo "## Knitting analysis.Rmd (bootstrap by question)..."
echo "   -> Outputs: analysis_by_question.html, .RData"
run Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}', output_dir='${output_dir}', output_name='analysis_by_question'), output_file='analysis_by_question.html', output_dir='${output_dir}')"

echo "## Knitting analysis.Rmd (bootstrap by user)..."
echo "   -> Outputs: analysis_by_user.html, .RData"
run Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}', output_dir='${output_dir}', output_name='analysis_by_user'), output_file='analysis_by_user.html', output_dir='${output_dir}')"

echo "## Knitting analysis.Rmd (OE on)..."
echo "   -> Outputs: analysis_OE_on.html, .RData"
run Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}', output_dir='${output_dir}', user_stratification='On OE', output_name='analysis_OE_on'), output_file='analysis_OE_on.html', output_dir='${output_dir}')"

echo "## Knitting analysis.Rmd (OE off)..."
echo "   -> Outputs: analysis_OE_off.html, .RData"
run Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}', output_dir='${output_dir}', user_stratification='Off OE', output_name='analysis_OE_off'), output_file='analysis_OE_off.html', output_dir='${output_dir}')"

# # Example code for running analysis where you drop only a single vendor from the analysis
# echo "## Knitting analysis.Rmd (bootstrap by question, drop vendor A)..."
# run Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}', output_dir='${output_dir}', drop_vendor='A', output_name='analysis_by_question_drop_A'), output_file='analysis_by_question_drop_A.html', output_dir='${output_dir}')"

echo "## Knitting exploratory_analysis.Rmd..."
echo "   -> Outputs: exploratory_analysis.html, .RData"
run Rscript -e "rmarkdown::render('exploratory_analysis.Rmd', params=list(data_dir='${data_dir}', output_dir='${output_dir}', output_name='exploratory_analysis'), output_file='exploratory_analysis.html', output_dir='${output_dir}')"
