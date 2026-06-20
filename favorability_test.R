get_favorability_score <- function(outcomes) {

  n <- length(outcomes)
  wins <- sum(outcomes == 1)
  losses <- sum(outcomes == -1)

  (wins - losses) / n
}

run_favorability_test <- function(df, n_boot = 1000) {

  stopifnot(all(c("question_id", "outcome") %in% names(df)))

  obs_fave <- get_favorability_score(df$outcome)

  questions <- unique(df$question_id)
  n_questions <- length(questions)

  null_faves <- replicate(n_boot, {
    sampled_questions <- sample(questions, n_questions, replace = TRUE)

    boot_outcomes <- unlist(lapply(sampled_questions, function(q) {
      q_outcomes <- df$outcome[df$question_id == q]
      sapply(q_outcomes, function(o) {
        if (o == 0) 0 else sample(c(-1, 1), 1)
      })
    }))

    get_favorability_score(boot_outcomes)
  })

  p_val <- mean(abs(null_faves) >= abs(obs_fave))
  p_val
}

simulate_null_data <- function(n_obs, null_dist) {
  counts <- as.vector(rmultinom(1, n_obs, null_dist))
  counts
}

get_favorability_score_from_counts <- function(empirical_dist) {
  (empirical_dist[1] - empirical_dist[3]) / sum(empirical_dist)
}

run_favorability_test_simple <- function(empirical_dist, n_boot_obs = 1000) {
  obs_fave <- get_favorability_score_from_counts(empirical_dist)

  n_obs <- sum(empirical_dist)
  null_distribution <- c(
    (empirical_dist[1] + empirical_dist[3]) / 2,
    empirical_dist[2],
    (empirical_dist[1] + empirical_dist[3]) / 2
  ) / n_obs

  all_null_faves <- replicate(n_boot_obs, {
    null_dist <- simulate_null_data(n_obs, null_distribution)
    get_favorability_score_from_counts(null_dist)
  })

  p_val <- mean(abs(all_null_faves) >= abs(obs_fave))
  p_val
}

calc_power <- function(n_obs, alt_distribution, alpha = 0.05,
                       num_replicates = 100, n_boot = 1000) {
  rejections <- replicate(num_replicates, {
    empirical_dist <- simulate_null_data(n_obs, alt_distribution)
    p_val <- run_favorability_test_simple(empirical_dist, n_boot_obs = n_boot)
    p_val < alpha
  })
  mean(rejections)
}
