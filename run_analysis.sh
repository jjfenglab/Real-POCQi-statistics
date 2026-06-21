#!/usr/bin/env bash
set -euo pipefail

# Usage: run_analysis.sh [data_dir] [mapping_path]
data_dir="${1:-pilot_data}"
mapping_path="${data_dir}/secret_mapping.txt"

echo "## Blinding vendor identities..."
Rscript blind_vendors.R "${data_dir}" "${mapping_path}"

echo "## Knitting descriptives.Rmd..."
Rscript -e "rmarkdown::render('descriptives.Rmd', params=list(data_dir='${data_dir}'), output_dir='${data_dir}')"

echo "## Knitting analysis.Rmd..."
Rscript -e "rmarkdown::render('analysis.Rmd', params=list(data_dir='${data_dir}'), output_dir='${data_dir}')"
