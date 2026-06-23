library(dplyr)

MIN_SUBMISSION_TIME_SECONDS <- 10

#' Calculate interrater agreement with bootstrap 95% CI
#'
#' Computes percent agreement and Fleiss' kappa for ratings of the same
#' question-vendor pair (regardless of slot order). Returns estimates with
#' bootstrap confidence intervals.
#'
#' @param qa_ratings Data frame from build_qa_ratings()
#' @param n_boot Number of bootstrap iterations
#' @param seed Random seed for reproducibility
#' @return List with overall stats and per-axis breakdown
calculate_interrater_agreement <- function(qa_ratings, n_boot = 1000, seed = 42) {
  set.seed(seed)

  raw_to_score <- c(strongly_a = 1, slightly_a = 2, tie = 3, slightly_b = 4, strongly_b = 5)

  standardized <- qa_ratings %>%
    mutate(
      vendor_pair = ifelse(slot_a_vendor < slot_b_vendor,
                           paste(slot_a_vendor, slot_b_vendor, sep = "_"),
                           paste(slot_b_vendor, slot_a_vendor, sep = "_")),
      slot_a_is_first = slot_a_vendor < slot_b_vendor,
      raw_score = raw_to_score[choice],
      score = ifelse(slot_a_is_first, raw_score, 6 - raw_score)
    )

  calc_agreement <- function(data) {
    pairs <- data %>%
      group_by(question_id, subversion, vendor_pair, axis) %>%
      filter(n() >= 2) %>%
      summarise(scores = list(score), .groups = "drop")

    if (nrow(pairs) == 0) {
      return(c(pct = NA_real_, n_pairs = 0))
    }

    n_agree <- 0
    n_total <- 0
    for (i in seq_len(nrow(pairs))) {
      s <- pairs$scores[[i]]
      n <- length(s)
      for (j in 1:(n - 1)) {
        for (k in (j + 1):n) {
          n_total <- n_total + 1
          if (abs(s[j] - s[k]) <= 1) n_agree <- n_agree + 1
        }
      }
    }
    c(pct = n_agree / n_total, n_pairs = nrow(pairs))
  }

  axes <- unique(standardized$axis)
  observed_per_axis <- lapply(axes, function(ax) {
    calc_agreement(standardized[standardized$axis == ax, ])
  })
  names(observed_per_axis) <- axes

  question_ids <- unique(standardized$question_id)
  n_questions <- length(question_ids)
  standardized_split <- split(standardized, standardized$question_id)

  boot_per_axis <- lapply(axes, function(ax) numeric(n_boot))
  names(boot_per_axis) <- axes

  for (b in seq_len(n_boot)) {
    boot_idx <- sample(question_ids, n_questions, replace = TRUE)
    boot_data <- do.call(rbind, standardized_split[boot_idx])

    for (ax in axes) {
      ax_data <- boot_data[boot_data$axis == ax, ]
      ax_stats <- calc_agreement(ax_data)
      boot_per_axis[[ax]][b] <- ax_stats["pct"]
    }
  }

  per_axis_results <- lapply(axes, function(ax) {
    list(
      axis = ax,
      pct_agreement = observed_per_axis[[ax]]["pct"],
      pct_ci = quantile(boot_per_axis[[ax]], c(0.025, 0.975), na.rm = TRUE),
      n_pairs = observed_per_axis[[ax]]["n_pairs"]
    )
  })
  names(per_axis_results) <- axes

  list(per_axis = per_axis_results)
}

apply_inclusion_criteria <- function(qa_ratings,
                                     min_submission_time = MIN_SUBMISSION_TIME_SECONDS) {
  qa_ratings %>%
    filter(!is.na(submission_time_seconds),
           submission_time_seconds >= min_submission_time)
}

get_eligible_users <- function(users, user_registrations = NULL, user_stratification = NULL) {
  test_email_patterns <- c("eval\\.test$", "example\\.com$", "openevidence\\.com$")
  eligible <- users %>%
    filter(subversion %in% c("qa_text_only", "qa_text_citations"),
           !grepl(paste(test_email_patterns, collapse = "|"), email, ignore.case = TRUE))

  if (!is.null(user_stratification) && !is.null(user_registrations)) {
    eligible_emails <- user_registrations %>%
      filter(`OE Status` == user_stratification) %>%
      pull(Email)
    eligible <- eligible %>% filter(email %in% eligible_emails)
  }

  eligible
}

build_qa_ratings <- function(ratings, assignments, questions, users,
                             prior_users = NULL) {
  if (!is.null(prior_users)) {
    n_dropped <- sum(unique(assignments$user_id) %in% prior_users$id)
    cat(sprintf("Dropped %d prior users from assignments\n", n_dropped))
    assignments <- assignments %>% filter(!user_id %in% prior_users$id)
  }

  eligible_users <- get_eligible_users(users)

  ratings %>%
    filter(!is.na(qa_assignment_id)) %>%
    inner_join(
      assignments %>%
        select(assignment_id = id, user_id, question_id,
               slot_a_vendor, slot_b_vendor, completed_at),
      by = c("qa_assignment_id" = "assignment_id")
    ) %>%
    inner_join(
      questions %>% select(question_id = id, specialty),
      by = "question_id"
    ) %>%
    inner_join(
      eligible_users %>% select(user_id = id, subversion),
      by = "user_id"
    ) %>%
    mutate(
      winner = case_when(
        choice %in% c("strongly_a", "slightly_a") ~ slot_a_vendor,
        choice %in% c("strongly_b", "slightly_b") ~ slot_b_vendor,
        choice == "tie" ~ NA_character_
      ),
      loser = case_when(
        choice %in% c("strongly_a", "slightly_a") ~ slot_b_vendor,
        choice %in% c("strongly_b", "slightly_b") ~ slot_a_vendor,
        choice == "tie" ~ NA_character_
      ),
      is_tie = choice == "tie"
    )
}
