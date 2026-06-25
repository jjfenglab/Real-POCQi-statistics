get_favorability_score <- function(outcomes, weights = NULL) {
  if (is.null(weights)) weights <- rep(1, length(outcomes))

  total_weight <- sum(weights)
  weighted_wins <- sum(weights[outcomes == 1])
  weighted_losses <- sum(weights[outcomes == -1])

  (weighted_wins - weighted_losses) / total_weight
}

run_favorability_test <- function(df, n_boot = 1000, ci_level = NULL,
                                  bootstrap_by = c("question", "user")) {

  bootstrap_by <- match.arg(bootstrap_by)
  cluster_col <- if (bootstrap_by == "question") "question_id" else "user_id"

  stopifnot(all(c(cluster_col, "outcome") %in% names(df)))

  weights <- if ("weight" %in% names(df)) df$weight else rep(1, nrow(df))
  obs_fave <- get_favorability_score(df$outcome, weights)

  outcome_splits <- split(df$outcome, df[[cluster_col]])
  weight_splits <- split(weights, df[[cluster_col]])
  n_clusters <- length(outcome_splits)

  null_faves <- replicate(n_boot, {
    idx <- sample.int(n_clusters, n_clusters, replace = TRUE)
    boot_outcomes <- unlist(outcome_splits[idx], use.names = FALSE)
    boot_weights <- unlist(weight_splits[idx], use.names = FALSE)
    flipped <- boot_outcomes * sample(c(-1, 1), length(boot_outcomes), replace = TRUE)
    sum(flipped * boot_weights) / sum(boot_weights)
  })

  p_val <- mean(abs(null_faves) >= abs(obs_fave))

  if (!is.null(ci_level)) {
    boot_faves <- replicate(n_boot, {
      idx <- sample.int(n_clusters, n_clusters, replace = TRUE)
      boot_outcomes <- unlist(outcome_splits[idx], use.names = FALSE)
      boot_weights <- unlist(weight_splits[idx], use.names = FALSE)
      get_favorability_score(boot_outcomes, boot_weights)
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

run_win_rate_ci <- function(df, n_boot = 1000, ci_level = 0.95,
                            bootstrap_by = c("question", "user"),
                            use_weights = FALSE) {

  bootstrap_by <- match.arg(bootstrap_by)
  cluster_col <- if (bootstrap_by == "question") "question_id" else "user_id"

  stopifnot(all(c(cluster_col, "row_win") %in% names(df)))
  stopifnot(ci_level > 0 && ci_level < 1)

  n_obs <- nrow(df)

  if (n_obs == 0) {
    return(list(
      win_rate = NA_real_,
      ci_lower = NA_real_,
      ci_upper = NA_real_,
      n = 0L,
      wins = 0L
    ))
  }

  if (use_weights && "weight" %in% names(df)) {
    weights <- df$weight
    total_weight <- sum(weights)
    weighted_wins <- sum(weights * df$row_win)
    rate_obs <- weighted_wins / total_weight
    wins_obs <- sum(df$row_win)

    win_splits <- split(as.integer(df$row_win), df[[cluster_col]])
    weight_splits <- split(weights, df[[cluster_col]])
    n_clusters <- length(win_splits)

    boot_rates <- replicate(n_boot, {
      idx <- sample.int(n_clusters, n_clusters, replace = TRUE)
      boot_wins <- unlist(win_splits[idx], use.names = FALSE)
      boot_weights <- unlist(weight_splits[idx], use.names = FALSE)
      total_w <- sum(boot_weights)
      if (total_w == 0) NA_real_ else sum(boot_weights * boot_wins) / total_w
    })
  } else {
    wins_obs <- sum(df$row_win)
    rate_obs <- wins_obs / n_obs

    splits <- split(as.integer(df$row_win), df[[cluster_col]])
    cluster_n    <- vapply(splits, length, integer(1))
    cluster_wins <- vapply(splits, sum, integer(1))
    n_clusters <- length(cluster_n)

    boot_rates <- replicate(n_boot, {
      idx <- sample.int(n_clusters, n_clusters, replace = TRUE)
      total_n    <- sum(cluster_n[idx])
      total_wins <- sum(cluster_wins[idx])
      if (total_n == 0) NA_real_ else total_wins / total_n
    })
  }

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
