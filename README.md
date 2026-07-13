# Statistical Analysis for "Expert Evaluation of Clinical AI Tools on Real Point-of-Care Clinical Queries"
Jean Feng, Vishal Patel, Patrick Heagerty, Yifan Mai, Venkatesh Sivaraman, Patrick Vossler, Jialin Ouyang, and Anupam B. Jena. 2026.

https://arxiv.org/abs/2606.28960

## Code requirements:
* Rscripts were run with R version 4.3.0
* Python requirements for running the sconscripts in `exp_themes/sconscript` (for finding question themes) are in requirements.txt. We recommend running the code in a python virtual environment created through pip.
* All code can be run on a standard laptop.

## Instructions for running software and reproducing analyses
* Statistical analyses are given in `run_analysis.sh`. These files will output both html files from the Rmd files and associated pdf files for the paper figures.
* Python analysis of question themes are run via `scons exp_themes`
* Run time should be no more than one hour.
