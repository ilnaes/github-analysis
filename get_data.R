library(dotenv)
library(httr)
library(jsonlite)
library(tidyverse)

token <- Sys.getenv("TOKEN")

GET("https://api.github.com/rate_limit", add_headers(Authorization = paste("token", token)))

gh_search <- function(lang, stars = 99999, auth = token) {
  endpoint <- "https://api.github.com/search/repositories"
  query <- paste0("?q=stars:<=", stars, "+language:", lang, "&sort=stars&order=desc&per_page=100")
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
    response <- gh_search(lang, stars)
    response <- extract_repos(response)
    
    if (is.null(response)) {
      print("BAD")
      Sys.sleep(10)
      next
    }
    
    res <- c(res, response)
    stars <- response[length(response)][[1]]$stargazers_count - 1
    print(paste(length(res), "---", stars))
    
    if (length(response) < 100 || length(res) >= n || stars < min) {
      break
    }
    Sys.sleep(1.2)
  }
  
  res
}

get_tibble <- function(response) {
  response |>
    map(\(x) do.call(tibble, list_modify(x,
                                         "owner" = x$owner$type,
                                         "license" = if (is.null(x$license)) {
                                           NA
                                         } else {
                                           paste(x$license, collapse = ";")
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


languages <- c("python", "js", "java", "go", "typescript", "cpp", "c", "matlab", "r", "jupyter-notebook")

dfs <- languages |> 
  map(\(x) get_tibble(get_repos(x, n = 1000)))

dfs |> 
  bind_rows() |>
  write_csv("raw.csv")

read_csv("raw.csv") |>
  mutate(language = ifelse(language == "MATLAB" | is.na(language), "Matlab", language)) |>
  select(-ends_with("url"), -node_id, -private, -fork, -disabled, -score, -stargazers_count, -watchers_count, -permissions) |>
  write_csv("clean.csv")