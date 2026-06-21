#!/usr/bin/env bash

echo "[1/2] Blinding vendor identities..."
Rscript blind_vendors.R

echo "[2/2] Knitting descriptives.Rmd..."
Rscript -e 'rmarkdown::render("descriptives.Rmd")'
