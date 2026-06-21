#!/usr/bin/env Rscript
# Reads raw pilot CSVs, remaps vendor identities to A/B/C/D using a random
# permutation, writes blinded_*.csv files that downstream analyses
# (descriptives.Rmd) consume, and writes secret_mapping.txt with the true
# identities. secret_mapping.txt MUST NOT be committed to version control.

suppressPackageStartupMessages(library(tidyverse))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript blind_vendors.R <data_dir> <mapping_path>")
}
data_dir     <- args[[1]]
mapping_path <- args[[2]]

assignments <- read_csv(file.path(data_dir, "qa_assignments.csv"),
                        show_col_types = FALSE)
questions   <- read_csv(file.path(data_dir, "qa_questions.csv"),
                        show_col_types = FALSE)
ratings     <- read_csv(file.path(data_dir, "ratings.csv"),
                        show_col_types = FALSE)
users       <- read_csv(file.path(data_dir, "users.csv"),
                        show_col_types = FALSE)
prior_users_path <- file.path(data_dir, "prior_users.csv")
prior_users <- if (file.exists(prior_users_path)) {
  read_csv(prior_users_path, show_col_types = FALSE)
} else {
  NULL
}

vendors <- sort(unique(c(assignments$slot_a_provider,
                         assignments$slot_b_provider)))

if (length(vendors) > length(LETTERS)) {
  stop("More vendors than available single-letter labels.")
}

vendor_mapping <- setNames(sample(LETTERS[seq_along(vendors)]), vendors)

assignments_blinded <- assignments %>%
  mutate(
    slot_a_vendor = unname(vendor_mapping[slot_a_provider]),
    slot_b_vendor = unname(vendor_mapping[slot_b_provider])
  ) %>%
  select(-slot_a_provider, -slot_b_provider)

write_csv(assignments_blinded,
          file.path(data_dir, "blinded_qa_assignments.csv"))
write_csv(questions, file.path(data_dir, "blinded_qa_questions.csv"))
write_csv(ratings,   file.path(data_dir, "blinded_ratings.csv"))
write_csv(users,     file.path(data_dir, "blinded_users.csv"))
if (!is.null(prior_users)) {
  write_csv(prior_users, file.path(data_dir, "blinded_prior_users.csv"))
}

writeLines(
  c(
    "# Secret vendor mapping -- DO NOT COMMIT",
    sprintf("# Generated: %s", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    "",
    sprintf("%s -> %s", names(vendor_mapping), unname(vendor_mapping))
  ),
  mapping_path
)

message(sprintf("Wrote blinded CSVs to %s/ and mapping to %s",
                data_dir, mapping_path))
