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

  calc_agreement <- function(data, id_col = "question_id") {
    pairs <- data %>%
      group_by(across(all_of(c(id_col, "subversion", "vendor_pair", "axis")))) %>%
      filter(n() >= 2) %>%
      summarise(scores = list(score), .groups = "drop")

    if (nrow(pairs) == 0) {
      return(list(pct = NA_real_, kappa = NA_real_, n_pairs = 0))
    }

    rater1 <- c()
    rater2 <- c()
    n_agree <- 0
    n_total <- 0
    for (i in seq_len(nrow(pairs))) {
      s <- pairs$scores[[i]]
      n <- length(s)
      for (j in 1:(n - 1)) {
        for (k in (j + 1):n) {
          n_total <- n_total + 1
          rater1 <- c(rater1, s[j])
          rater2 <- c(rater2, s[k])
          if (abs(s[j] - s[k]) <= 1) n_agree <- n_agree + 1
        }
      }
    }

    kappa_val <- irr::kappa2(cbind(rater1, rater2), weight = "squared")$value
    list(pct = n_agree / n_total, kappa = kappa_val, n_pairs = nrow(pairs))
  }

  axes <- unique(standardized$axis)
  observed_per_axis <- lapply(axes, function(ax) {
    calc_agreement(standardized[standardized$axis == ax, ])
  })
  names(observed_per_axis) <- axes

  question_ids <- unique(standardized$question_id)
  n_questions <- length(question_ids)
  standardized_split <- split(standardized, standardized$question_id)

  boot_per_axis <- lapply(axes, function(ax) list(pct = numeric(n_boot), kappa = numeric(n_boot)))
  names(boot_per_axis) <- axes

  for (b in seq_len(n_boot)) {
    boot_idx <- sample(question_ids, n_questions, replace = TRUE)
    boot_data_list <- lapply(seq_along(boot_idx), function(i) {
      df <- standardized_split[[as.character(boot_idx[i])]]
      df$boot_question_id <- paste(boot_idx[i], i, sep = "_")
      df
    })
    boot_data <- do.call(rbind, boot_data_list)

    for (ax in axes) {
      ax_data <- boot_data[boot_data$axis == ax, ]
      ax_stats <- calc_agreement(ax_data, id_col = "boot_question_id")
      boot_per_axis[[ax]]$pct[b] <- ax_stats$pct
      boot_per_axis[[ax]]$kappa[b] <- ax_stats$kappa
    }
  }

  per_axis_results <- lapply(axes, function(ax) {
    list(
      axis = ax,
      pct_agreement = observed_per_axis[[ax]]$pct,
      pct_ci = quantile(boot_per_axis[[ax]]$pct, c(0.025, 0.975), na.rm = TRUE),
      kappa = observed_per_axis[[ax]]$kappa,
      kappa_ci = quantile(boot_per_axis[[ax]]$kappa, c(0.025, 0.975), na.rm = TRUE),
      n_pairs = observed_per_axis[[ax]]$n_pairs
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

  in_experiment <- users %>%
    filter(subversion %in% c("qa_text_only", "qa_text_citations"))

  n_fake <- sum(grepl(paste(test_email_patterns, collapse = "|"), in_experiment$email, ignore.case = TRUE))
  cat(sprintf("Excluded %d users with test/example email addresses\n", n_fake))

  eligible <- in_experiment %>%
    filter(!grepl(paste(test_email_patterns, collapse = "|"), email, ignore.case = TRUE))

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
