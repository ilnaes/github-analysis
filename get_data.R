library(dotenv)
library(httr)
library(jsonlite)
library(tidyverse)

token <- Sys.getenv("TOKEN")

GET("https://api.github.com/rate_limit", add_headers(Authorization = paste("token", token)))

gh_search <- function(lang, stars = 99999, page = 1, auth = token) {
  endpoint <- "https://api.github.com/search/repositories"
  query <- paste0("?q=stars:<=", stars, "+language:", lang, "&sort=stars&order=desc&per_page=100&page=", page)
  token <- add_headers(Authorization = paste("token", auth))

  GET(paste0(endpoint, query), token)
}

extract_repos <- function(response) {
  if (status_code(response) != 200) {
    return(NULL)
  }

  res <- parse_json(rawToChar(response$content))
  res$items
}

get_repos <- function(lang, n = 1000, stars = 99999, min = 10) {
  print(lang)
  res <- list()

  while (TRUE) {
    p <- 1
    
    while (p <= 10) {
      response <- gh_search(lang, stars, p)
      response <- extract_repos(response)
  
      if (is.null(response)) {
        print("BAD")
        Sys.sleep(10)
        next
      }

      res <- c(res, response)
      print(paste0(length(res), ":", p, " --- ", response[length(response)][[1]]$stargazers_count))
  
      if (length(response) < 100 || length(res) >= n || stars < min) {
        return(res)
      }
      
      p <- p + 1
      Sys.sleep(1.2)
    }
    
    stars <- response[length(response)][[1]]$stargazers_count - 1
  }
}

get_tibble <- function(response) {
  response |>
    map(\(x) do.call(tibble, list_modify(x,
      "owner" = x$owner$type,
      "license" = if (is.null(x$license)) {
        NA
      } else {
        x$license$name
      },
      "mirror_url" = NULL,
      "permissions" = if (is.null(x$permissions)) {
        NA
      } else {
        paste(x$permission, collapse = ";")
      },
    ))) |>
    bind_rows()
}

languages <- c("python", "js", "java", "c", "r")
num <- c(4000, 4000, 4000, 4000, 4000)

dfs <- map2(languages, num,
            \(x, y)
            get_tibble(get_repos(x, n = y)) |> 
              distinct(full_name, .keep_all = TRUE) |> 
              mutate(language = x))

dfs |>
  bind_rows() |>
  write_csv("data/raw.csv")

read_csv("data/raw.csv") |>
  select(-ends_with("url"), -node_id, -private, -fork, -disabled, -score, -stargazers_count, -watchers_count, -permissions) |>
  rename(stars = watchers) |> 
  write_csv("data/clean.csv")
