get_favorability_score <- function(outcomes) {

  n <- length(outcomes)
  wins <- sum(outcomes == 1)
  losses <- sum(outcomes == -1)

  (wins - losses) / n
}

run_favorability_test <- function(df, n_boot = 1000, ci_level = NULL) {

  stopifnot(all(c("question_id", "outcome") %in% names(df)))

  obs_fave <- get_favorability_score(df$outcome)

  splits <- split(df$outcome, df$question_id)
  n_questions <- length(splits)

  null_faves <- replicate(n_boot, {
    idx <- sample.int(n_questions, n_questions, replace = TRUE)
    boot_outcomes <- unlist(splits[idx], use.names = FALSE)
    flipped <- boot_outcomes * sample(c(-1, 1), length(boot_outcomes), replace = TRUE)
    mean(flipped)
  })

  p_val <- mean(abs(null_faves) >= abs(obs_fave))

  if (!is.null(ci_level)) {
    boot_faves <- replicate(n_boot, {
      idx <- sample.int(n_questions, n_questions, replace = TRUE)
      mean(unlist(splits[idx], use.names = FALSE))
    })

    alpha <- 1 - ci_level
    ci <- quantile(boot_faves, c(alpha / 2, 1 - alpha / 2), names = FALSE)
  } else {
    ci <- c(NULL, NULL)
  }

  list(
    p_value = p_val,
    favorability = obs_fave,
    ci_lower = ci[1],
    ci_upper = ci[2]
  )
}

run_win_rate_ci <- function(df, n_boot = 1000, ci_level = 0.95) {

  stopifnot(all(c("question_id", "row_win") %in% names(df)))
  stopifnot(ci_level > 0 && ci_level < 1)

  n_obs <- nrow(df)
  wins_obs <- sum(df$row_win)
  rate_obs <- if (n_obs > 0) wins_obs / n_obs else NA_real_

  if (n_obs == 0) {
    return(list(
      win_rate = NA_real_,
      ci_lower = NA_real_,
      ci_upper = NA_real_,
      n = 0L,
      wins = 0L
    ))
  }

  splits <- split(as.integer(df$row_win), df$question_id)
  q_n    <- vapply(splits, length, integer(1))
  q_wins <- vapply(splits, sum, integer(1))
  n_questions <- length(q_n)

  boot_rates <- replicate(n_boot, {
    idx <- sample.int(n_questions, n_questions, replace = TRUE)
    total_n    <- sum(q_n[idx])
    total_wins <- sum(q_wins[idx])
    if (total_n == 0) NA_real_ else total_wins / total_n
  })

  alpha <- 1 - ci_level
  ci <- quantile(boot_rates, c(alpha / 2, 1 - alpha / 2),
                 names = FALSE, na.rm = TRUE)

  list(
    win_rate = rate_obs,
    ci_lower = ci[1],
    ci_upper = ci[2],
    n = n_obs,
    wins = as.integer(wins_obs)
  )
}

calc_power <- function(n_obs, alt_distribution, alpha = 0.05,
                       num_replicates = 100, n_boot = 1000) {
  rejections <- replicate(num_replicates, {
    counts <- as.vector(rmultinom(1, n_obs, alt_distribution))
    outcomes <- rep(c(1, 0, -1), counts)
    df <- data.frame(question_id = seq_along(outcomes), outcome = outcomes)
    result <- run_favorability_test(df, n_boot = n_boot, ci_level = NULL)
    result$p_value < alpha
  })
  mean(rejections)
}
