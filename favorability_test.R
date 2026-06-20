simulate_null_data <- function(n_obs, null_dist) {
  counts <- as.vector(rmultinom(1, n_obs, null_dist))
  counts
}

get_favorability_score <- function(empirical_dist) {
  (empirical_dist[1] - empirical_dist[3]) / sum(empirical_dist)
}

run_favorability_test <- function(empirical_dist, n_boot_obs = 1000) {
  obs_fave <- get_favorability_score(empirical_dist)

  n_obs <- sum(empirical_dist)
  null_distribution <- c(
    (empirical_dist[1] + empirical_dist[3]) / 2,
    empirical_dist[2],
    (empirical_dist[1] + empirical_dist[3]) / 2
  ) / n_obs

  all_null_faves <- replicate(n_boot_obs, {
    null_dist <- simulate_null_data(n_obs, null_distribution)
    get_favorability_score(null_dist)
  })

  p_val <- mean(abs(all_null_faves) >= abs(obs_fave))
  p_val
}

calc_power <- function(n_obs, alt_distribution, alpha = 0.05,
                       num_replicates = 100, n_boot = 1000) {
  rejections <- replicate(num_replicates, {
    empirical_dist <- simulate_null_data(n_obs, alt_distribution)
    p_val <- run_favorability_test(empirical_dist, n_boot_obs = n_boot)
    p_val < alpha
  })
  mean(rejections)
}
