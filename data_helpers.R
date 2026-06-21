library(dplyr)

MIN_SUBMISSION_TIME_SECONDS <- 10

apply_inclusion_criteria <- function(qa_ratings,
                                     min_submission_time = MIN_SUBMISSION_TIME_SECONDS) {
  qa_ratings %>%
    filter(!is.na(submission_time_seconds),
           submission_time_seconds >= min_submission_time)
}

build_qa_ratings <- function(ratings, assignments, questions, users,
                             prior_users = NULL) {
  if (!is.null(prior_users)) {
    n_dropped <- sum(unique(assignments$user_id) %in% prior_users$id)
    cat(sprintf("Dropped %d prior users from assignments\n", n_dropped))
    assignments <- assignments %>% filter(!user_id %in% prior_users$id)
  }
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
      users %>% select(user_id = id, subversion),
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
