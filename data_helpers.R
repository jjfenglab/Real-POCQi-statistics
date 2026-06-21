library(dplyr)

build_qa_ratings <- function(ratings, assignments, questions, users,
                             prior_users = NULL) {
  if (!is.null(prior_users)) {
    assignments <- assignments %>% filter(!user_id %in% prior_users)
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
