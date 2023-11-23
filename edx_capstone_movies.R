# load dependencies
packs <- c(
  # plotting, grammar
  "tidyverse", "ggthemes", "colorspace", "gganimate",
  # modeling tools:
   "locfit", "mgcv", "caret",
  # misc utilities
  "ggpubr", "gridExtra", "data.table", "strex", "reshape", "reshape2")

# function to load packages and install as needed
loadpack <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
   if (length(new.pkg))
     install.packages(new.pkg, dependencies = TRUE)
   sapply(pkg, require, character.only = TRUE)
}

suppressMessages(loadpack(packs))
rm(loadpacks)

# histogram fill color
hist.fill = "#648FFF"
norm.fill = "#FFC107"


#  Helper functions ====   
extract_intl_title <- function(string){
  # we want whatever is in parentheses
  # before the release year that isn't 
  # 12 monkeys, Se7en or an a.k.a|A.K.A|aka|AKA
  if({str_count(string, "\\(") >= 2 & 
      !str_detect(string, "aka|a.k.a.|a.k.a|AKA|A.K.A|A.K.A.|12|Twelve|Se7en")}){
    
    new.string = str_extract(string, "^(.*)\\s*\\((.*)\\)\\s*(\\(\\d{4}\\))$", group = 2)
    return(new.string)
  }
  else{return("none")}
}
extract_year <- function(string){
  pattern = "\\((\\d{4})\\)"
  if(str_detect(string, pattern)){
    year = str_extract(string, pattern, group = 1) %>% parse_number()
    return(year)
  }
  else{
    return(0)
  }
}
process_title <- function(string){
  # This function extracts everything before the first open parentheses 
  # and re-arranges the title if necessary.
  
  
  # If there's any parenthesed data, extract everything
  # before the first open parentheses
  if(str_detect(string, "\\(")){
    string = strex::str_before_first(string, "\\(") %>% str_trim()
  }
  ## Re-Arrange the title if necessary
  
  # Some int'l films are recorded as "Enfant, Le'"
  if(str_detect(string, ",\\s*\\w+'$")){
    new_string = str_replace_all(string, "^(.*),(.*)$","\\2\\1")
    return(str_trim(new_string))
  }
  # Some films recorded as "Hero, The"
  else if(str_detect(string, ",\\s*\\w+$")){
    new_string = str_replace_all(string, "^(.*),(.*)$","\\2 \\1")
    return(str_trim(new_string))
  }
  else{
    return(string)
  }
} 
## RMSE
get.rmse <- function(r, rhat){return(sqrt(mean((r-rhat)^2, na.rm=T)))}
rmse.df <- function(var){return(sqrt(mean(var^2, na.rm=T)))}

# Load Files ==== 
ratings.file <- "ml-10M100K/ratings.dat"
titles.file <- "ml-10M100K/movies.dat"

# Read in ratings ====  
ratings <- read_delim(ratings.file, 
                      delim = "::", 
                      col_names = c("userID", "movieID", "rating", "timestamp"),
                      col_types = "iidi") %>%
  as.data.table(key = c("userID","movieID")) %>%
  mutate(timestamp = lubridate::as_datetime(timestamp, origin = "1970-01-01"),
         rev.year  = lubridate::year(timestamp),
         rev.month = lubridate::month(timestamp),
         rev.day   = lubridate::day(timestamp),
         rev.week  = lubridate::week(timestamp),
         rev.weekday = lubridate::wday(timestamp))

gc()

# Read in titles ====  
titles <-
  read_delim(titles.file, delim = "::",
             col_names = c("movieID", "title", "genre"),
             col_types = "icc") %>%
  as.data.table(key = c("movieID")) %>%
  mutate(year = sapply(title, extract_year),
         intl.title = sapply(title, extract_intl_title),
         title = sapply(title, process_title),
         genre = ifelse(str_count(genre, "\\|") < 4,
                        genre,
                        ## by keeping everything up to 4th '|'
                        strex::str_before_nth(genre, "\\|", 4))) %>%
  filter(movieID %in% ratings$movieID)
gc()


# Join titles and ratings ====
movielens <- ratings[titles, on = "movieID"]

# Partition Raw Data ====  
suppressWarnings(set.seed(1, sample.kind = "Rounding"))
test.idx <- caret::createDataPartition(movielens$rating, 
                                       times = 1, 
                                       p = 0.1, 
                                       list = F)
temp <- movielens[test.idx]
edx  <- movielens[-test.idx]
rm(test.idx)
gc()

## UserID & MovieID need to be in training and testing
suppressMessages(
  final_holdout_test <- 
  temp %>%
  semi_join(edx, by = "movieID") %>%
  semi_join(edx, by = "userID")
)
suppressMessages(
  removed <- anti_join(temp, final_holdout_test)
)
suppressMessages(
  edx <- rbind(edx, removed)
)
# order by movieID and userID
edx = edx[order(movieID, userID)]
final_holdout_test = final_holdout_test[order(movieID, userID)]

# clear junk
rm(ratings, titles, movielens, temp, removed, 
   process_title, extract_intl_title, extract_year); 
invisible(gc())

# Partition Into Test & Train ====
# use 90% for training and 10% for testing

suppressWarnings(set.seed(1936, sample.kind = "Rounding"))
idx = caret::createDataPartition(edx$rating, p = 0.10, list = F)
test  = edx[idx, ]
train = edx[-idx, ]


# clear memory
rm(idx, ratings.file, titles.file)
invisible(gc())

# *** Question ***
# -- What combination of regularization, & minimum number of reviews for users and movies
#    leads to to the best cross validated performance? 

lambda.grid= expand.grid(
  lambda.u = c(0:5),
  lambda.m = c(0:5),
  nrev.u   = c(0, 10, 15, 30),
  nrev.m  = c(0, 10, 15, 30),
  train.c    = 0, 
  test.c     = 0,
  train.s    = 0, 
  test.s     = 0,
  stringsAsFactors = F
)

start = Sys.time()
for(i in 1:nrow(lambda.grid)){
  # train the model
  dat =
    train[, ':='(centered = rating-mean(rating),
                 scaled   =(rating-mean(rating))/sd(rating))
          ][,nrev.m := .N, by = "movieID"
            ][nrev.m >= lambda.grid$nrev.m[i]
              ][,nrev.u := .N, by = "userID"
                ][nrev.u >= lambda.grid$nrev.u[i]
                  ][, ':=' (movie.effect.c = sum(centered)/(.N + lambda.grid$lambda.m[i]),
                            movie.effect.s = sum(scaled)/(.N + lambda.grid$lambda.m[i]))
                    , by = "movieID"
                    ][,':='(residual.c = centered-movie.effect.c,
                            residual.s = centered-movie.effect.s)
                      ][,':='(user.effect.c = sum(residual.c)/(.N + lambda.grid$lambda.u[i]),
                              user.effect.s = sum(residual.s)/(.N + lambda.grid$lambda.u[i])),
                        by = "userID"
                        ][,':='(residual.c = residual.c-movie.effect.c,
                                residual.s = residual.s-movie.effect.s)]
  
  # extract MU & SD
  MU = mean(dat$rating)
  SD = sd(dat$rating)
  
  # +++++++++++++++++++++++++++ #
  # Calculate train rmse 
  # +++++++++++++++++++++++++++ #
  # centered train rmse
  lambda.grid$train.c[i] = rmse.df(dat$residual.c)
  # scaled train rmse
  lambda.grid$train.s[i] = rmse.df(dat$residual.s)
  
  
  
  # +++++++++++++++++++++++++++ #
  # Extract Effects 
  # +++++++++++++++++++++++++++ #
  
  # centered effects
  users.c  = unique(dat[,c('userID','user.effect.c')],  by = c('userID','user.effect.c'))
  movies.c = unique(dat[,c('movieID','movie.effect.c')],by = c('movieID','movie.effect.c'))
  
  u.effect.c = mean(users.c$user.effect.c)
  m.effect.c = mean(movies.c$movie.effect.c)
  # scaled effects
  users.s  = unique(dat[,c('userID','user.effect.s')],  by = c('userID','user.effect.s'))
  movies.s = unique(dat[,c('movieID','movie.effect.s')],by = c('movieID','movie.effect.s'))
  
  u.effect.s = mean(users.s$user.effect.s)
  m.effect.s = mean(movies.s$movie.effect.s)
  
  # +++++++++++++++++++++++++++ #
  # Join Effects to Test Data 
  # +++++++++++++++++++++++++++ #
  
  # add effects to test
  val <-
    test[,':='(centered =  rating-mean(dat$rating),
               scaled   = (rating-mean(dat$rating))/sd(dat$rating))
         ][ ,nrev.m := .N, by = "movieID"
           # keep movies with more than x reviews
           ][nrev.m >= lambda.grid$nrev.m[i]
             # number of ratings per user
             ][ ,nrev.u := .N, by = "userID"
               # keep users with more than x reviews
               ][nrev.u >= lambda.grid$nrev.u[i]
                 # join centered effects 
                 ][users.c, on = "userID"
                   ][movies.c, on = "movieID"
                     # Join scaled effects
                     ][users.s, on = "userID"
                       ][movies.s, on = "movieID"
                         # adjust for NAs
                         ][, ':=' (user.effect.c = ifelse(is.na(user.effect.c), u.effect.c, user.effect.c),
                                   user.effect.s = ifelse(is.na(user.effect.s), u.effect.s, user.effect.s))
                           # calculate the residual
                           ][,':='(residual.c = centered - user.effect.c - movie.effect.c,
                                   residual.s = scaled   - user.effect.s - movie.effect.s)]
  
  # +++++++++++++++++++++++++++ #
  # Calculate Test rmse 
  # +++++++++++++++++++++++++++ #
  
  lambda.grid$test.c[i] = rmse.df(val$residual.c)
  lambda.grid$test.s[i] = rmse.df(val$residual.s)
  
  # +++++++++++++++++++++++++++ #
  # clear items and memory
  # +++++++++++++++++++++++++++ #
  # get rid of everything
  rm(dat, val, 
     users.c, movies.c, users.s, movies.s,
     u.effect.c, u.effect.s, m.effect.s, m.effect.c)
  gc()
  
}


# Final Effects

# stats
lambda = 5
MU = mean(train$rating)
SD = sd(train$rating)

# train effects ====

# train.dt ====
train <- 
  ## filter out movies made before 1930
  # and ratings submitted prior to 1996
  train[year >= 1930 & rev.year >= 1996
        # normalize the ratings
        ][ ,norm.rating := (rating-MU)/SD
           # number of ratings per movie
           ][,nrev.m :=.N, by = "movieID" 
             # require min 20 ratings/movie
             ][nrev.m >= 20
               # calculate movie effects
               ][,movie.effect := sum(norm.rating)/(.N+lambda), by = "movieID"
                 # remove movie effect
                 ][,residual := norm.rating-movie.effect
                   # number of ratings per user
                   ][,nrev.u := .N, by = "userID"
                     # require min 20 ratings/user
                     ][nrev.u >= 20
                       # calculate user effect
                       ][,user.effect := sum(residual)/(.N + lambda), by = "userID"
                         # remove the user effect
                         ][,residual := norm.rating-movie.effect-user.effect
                           ]
  
# extract train effects ====
users  = unique(train[,c('userID','user.effect')],
                   by = c('userID','user.effect')) 

movies = unique(train[,c('movieID','movie.effect')],
                   by = c('movieID','movie.effect'))


# join effects to test data
test <-
  # filter out movies made before 1930
  # and ratings submitted prior to 1996
  test[year >= 1930 & rev.year >= 1996
       # normalize the ratings
       ][,norm.rating := (rating-MU)/SD
         # number of ratings per movie
         ][,nrev.m := .N, by = "movieID"
           # keep movies with 20 or more ratings
           ][nrev.m>=20
             # number of ratings per user
             ][,nrev.u := .N, by = "userID"
               # keep users with 20 or more ratings
               ][nrev.u >= 20
                 # add user and movie effects 
                 ][users, on = "userID"
                   ][movies, on = "movieID"
                     # calculate the residual
                     ][,residual := norm.rating-movie.effect-user.effect]
# clear junk
rm(movies, users, MU, SD)
invisible(gc())


# EDX & Holdout
MU = mean(edx$rating)
SD = sd(edx$rating)
start = now()
edx <-
  # filter out movies made before 1930
  # and ratings submitted prior to 1996
  edx[year >= 1930 & rev.year >= 1996
        # normalize the ratings
        ][ ,norm.rating := (rating-MU)/SD
           # number of ratings per movie
           ][,nrev.m :=.N, by = "movieID" 
             # require min 20 ratings/movie
             ][nrev.m >= 20
               # calculate movie effects
               ][,movie.effect := sum(norm.rating)/(.N+lambda), by = "movieID"
                 # remove movie effect
                 ][,residual := norm.rating-movie.effect
                   # number of ratings per user
                   ][,nrev.u := .N, by = "userID"
                     # require min 20 ratings/user
                     ][nrev.u >= 20
                       # calculate user effect
                       ][,user.effect := sum(residual)/(.N + lambda), by = "userID"
                         # remove the user effect
                         ][,residual := norm.rating-movie.effect-user.effect
                           ]

# extract the effects
users  = unique(edx[,c('userID','user.effect')],  by = c('userID','user.effect'))
movies = unique(edx[,c('movieID','movie.effect')],by = c('movieID','movie.effect'))

u.effect = mean(users$user.effect)
m.effect = mean(movies$movie.effect)

# join the effects to the holdout set
final_holdout_test <-
  # filter out movies made before 1930
  # and ratings submitted prior to 1996
  final_holdout_test[year >= 1930 & rev.year >= 1996
       # normalize the ratings
       ][ ,norm.rating := (rating-MU)/SD
         # number of ratings per movie
         ][ ,nrev.m := .N, by = "movieID"
           # keep movies with 20 or more ratings
           ][nrev.m>=20
             # number of ratings per user
             ][ ,nrev.u := .N, by = "userID"
               # keep users with 20 or more ratings
               ][nrev.u >= 20
                 # add user and movie effects 
                 ][users, on = "userID"
                   ][movies, on = "movieID"
                     # adjust for NAS
                   ][, user.effect := ifelse(is.na(user.effect), u.effect, user.effect)
                     ][ , movie.effect := ifelse(is.na(movie.effect), m.effect, movie.effect)
                        # calculate residual
                        ][,residual := norm.rating-user.effect-movie.effect]
# clear junk
rm(movies, users, lambda, MU, SD, start, end)
invisible(gc())

# TRAIN & TEST ====

# Quantiles 1  ====
quants <-
  quantile(train[,.N, by = movieID] %>% pull(N),
           probs = c(0.25, 0.50, 0.75))

# 25th, 50th, & 75 quantiles
q1 = quants[[1]]
q2 = quants[[2]]
q3 = quants[[3]]
# Engineer: Train & Test ====

train <- 
  train[, rank.m := 
          fcase(
    nrev.m <  q1,                1,
    nrev.m >= q1 & nrev.m < q2, 2,
    nrev.m >= q2 & nrev.m < q3, 3,
    nrev.m >= q3               , 4
  )
  ][,m.rank := factor(rank.m, ordered = T)
    ][, era := factor(
      fcase(
        year >= 1930 & year < 1948,  "Golden Age",
        year >= 1948 & year < 1965,  "Fall of the Studio",
        year >= 1965 & year < 1975,  "New Era",
        year >= 1975 & year < 1983,  "New & Blockbuster",
        year >= 1983              ,  "Blockbuster Age"),
      levels = c("Golden Age", "Fall of the Studio", "New Era",
                 "New & Blockbuster", "Blockbuster Age"),
      ordered = TRUE)
    ]

test <-
  test[, rank.m := 
         fcase(
           nrev.m <  q1,               1,
           nrev.m >= q1 & nrev.m < q2, 2,
           nrev.m >= q2 & nrev.m < q3, 3,
           nrev.m >= q3              , 4)
       ][, m.rank := factor(rank.m, ordered=T)
         ][, era := factor(
           fcase(
             year >= 1930 & year < 1948,  "Golden Age",
             year >= 1948 & year < 1965,  "Fall of the Studio",
             year >= 1965 & year < 1975,  "New Era",
             year >= 1975 & year < 1983,  "New & Blockbuster",
             year >= 1983              ,  "Blockbuster Age"),
           levels = c("Golden Age", "Fall of the Studio", "New Era",
                      "New & Blockbuster", "Blockbuster Age"),
           ordered = TRUE)
           ]

# memory 1 ====
rm(quants, q1, q2, q3)
invisible(gc())

# EDX & HOLDOUT ====
# Quantiles 2 ====
quants <-
  quantile(edx[,.N, by = movieID] %>% pull(N),
           probs = c(0.25, 0.50, 0.75))

# 25th, 50th, & 75 quantiles
q1 = quants[[1]]
q2 = quants[[2]]
q3 = quants[[3]]

# Engineer: Edx & Holdout ====
edx <- 
  edx[, rank.m := fcase(
    nrev.m <  q1,               1,
    nrev.m >= q1 & nrev.m < q2, 2,
    nrev.m >= q2 & nrev.m < q3, 3,
    nrev.m >= q3              , 4
  )
  ][, era := factor(
    fcase(
      year >= 1930 & year < 1948,  "Golden Age",
      year >= 1948 & year < 1965,  "Fall of the Studio",
      year >= 1965 & year < 1975,  "New Era",
      year >= 1975 & year < 1983,  "New & Blockbuster",
      year >= 1983              ,  "Blockbuster Age"),
    levels = c("Golden Age", "Fall of the Studio", "New Era",
               "New & Blockbuster", "Blockbuster Age"),
    ordered = TRUE)
  ]

final_holdout_test <-
  final_holdout_test[, nrev.m := .N, by = movieID
                     ][, nrev.u := .N, by = userID
                       ][,rank.m := fcase(
                         nrev.m <  q1,               1,
                         nrev.m >= q1 & nrev.m < q2, 2,
                         nrev.m >= q2 & nrev.m < q3, 3,
                         nrev.m >= q3              , 4
  )][, era := factor(
    fcase(
      year >= 1930 & year < 1948,  "Golden Age",
      year >= 1948 & year < 1965,  "Fall of the Studio",
      year >= 1965 & year < 1975,  "New Era",
      year >= 1975 & year < 1983,  "New & Blockbuster",
      year >= 1983              ,  "Blockbuster Age"),
    levels = c("Golden Age", "Fall of the Studio", "New Era",
               "New & Blockbuster", "Blockbuster Age"),
    ordered = TRUE)
  ]

# memory 2 ====
rm(quants, q1, q2, q3)
invisible(gc())

test.na <- test %>% filter(is.na(user.effect))
test <- test %>% filter(!is.na(user.effect))

# ********************** #
# Split Data Models ====
# ********************** #

## Split Train & Test by Rank
train.split = split(train[order(rank.m)], by = "rank.m")
test.split  = split(test[order(rank.m)], by = "rank.m")
gc()


# ******************** #
# GAM: Split Data ====
# ******************** #
## knots
knts.era = list(year = c(1930, 1948, 1965, 1975, 1983, 2008))
## grid
gam.grid = expand.grid(rank = 1:4,
                       ks = 5:30,
                       fit.time = 0,
                       predict.time = 0,
                       train.rmse = 0,
                       test.rmse = 0) %>% 
  arrange(rank, ks)
## time

## counters
rank = 1
n.iter = 1
## loop
train.split = lapply(train.split,\(train.data){
  # identify the training set
  test.data = test.split[[rank]]
  # inform user
  # K's to test ====     
    ks = 5:30
    # evaluate each K ====
    for(cp in ks){
      fit.time = now()
      #####################
      ## Fitting
      fit = mgcv::bam(residual ~ s(year, k = cp), 
                      data = train.data,
                      knots = knts.era, 
                      method="REML")
      # fitting time
      gam.grid$fit.time[n.iter] <<- difftime(now(), fit.time, units="secs")[[1]] %>% round(2)
      rm(fit.time)
       
      #####################
      ## Predicting
      pred.time = now()
      train.preds = predict(fit, train.data)
      test.preds  = predict(fit, test.data)
      ## prediction time
      gam.grid$predict.time[n.iter] <<-  difftime(now(), pred.time, units="secs")[[1]] %>% round(2)
      rm(pred.time)
      ############################
      ## Add To Tables
      ## Train
      train.data[[glue::glue("split.gam{cp}")]] = train.preds
      
      ## Test
      if(length(test.preds) == nrow(test.data)){
        empty.dt  = data.table(matrix(nrow = nrow(test.data)))
        empty.dt[[glue::glue("split.gam{cp}")]] = test.preds
        empty.dt = empty.dt[,2]
        temp = list(test.split[[rank]], empty.dt)
        test.split[[rank]] <<- setDT(unlist(temp, recursive = F), 
                                     check.names = T)
        rm(temp, empty.dt)
      }
      if(length(test.preds) != nrow(test.data)){
        print("incompatible vector length")
      }
     
      ##################
      ## RMSE
      gam.grid$train.rmse[n.iter] <<- get.rmse(train.data$residual, train.preds)
      gam.grid$test.rmse[n.iter]  <<- get.rmse(test.data$residual, test.preds)

      # clear junk
      rm(fit, train.preds, test.preds)
      gc()

      # increment the model counter
      n.iter <<- n.iter + 1
    }
    # increment the rank counter
    rank <<- rank + 1
    # return the data frame
    train.data
  })
print("done validating all K's for each rank")


# ************************* #
# Loess:: Split Data ====
# ************************* #

loess.grid  = expand.grid(rank = 1:4,
                          span = c(.10,.15,.20,.25,.30,.40),
                          fit.time = 0,
                          predict.time = 0,
                          train.rmse = 0,
                          test.rmse = 0) %>% 
  arrange(rank, span)

rank = 1
n.iter = 1
# start lapply
train.split <- lapply(train.split, \(train.data){
  # identify test set
  test.data = test.split[[rank]]
  # specify spans
  spans = c(.10,.15,.20,.25,.30,.40)
  # validate spans
  for(span in spans){
    #####################
    ## Fitting
    fit.time   = now()
    fit = locfit(residual ~ lp(year, nn = span, deg = 3), 
                   data = train.data)
    # fit time
    loess.grid$fit.time[n.iter] <<- difftime(now(),fit.time, units = "secs")[[1]] %>% round(2)
    rm(fit.time)
    #####################
    ## Predicting
    pred.time = now()
    train.preds = predict(fit, train.data)
    test.preds  = predict(fit, test.data)
    # predict time
    loess.grid$predict.time[n.iter] <<- difftime(now(),pred.time, units = "secs")[[1]] %>% round(2)
    rm(pred.time)
    ############################
    ## Add To Tables
    ## Train
    train.data[[glue::glue("split.loe{span}")]] = train.preds
    
    ## Test
     if(length(test.preds) == nrow(test.data)){
        empty.dt  = data.table(matrix(nrow = nrow(test.data)))
        empty.dt[[glue::glue("split.loe{span}")]] = test.preds
        empty.dt = empty.dt[,2]
        temp = list(test.split[[rank]], empty.dt)
        test.split[[rank]] <<- setDT(unlist(temp, recursive = F), 
                                     check.names = T)
        rm(empty.dt, temp)
      }
      if(length(test.preds) != nrow(test.data)){
        print("incompatible vector length")
      }
    
    # *******************
    # RMSE
    loess.grid$train.rmse[n.iter] <<- get.rmse(train.data$residual, train.preds)
    loess.grid$test.rmse[n.iter]  <<- get.rmse(test.data$residual, test.preds)
    
    # *******************
    # Junk Removal
    rm(fit, train.preds, test.preds)
    gc()
    
    # increment model counter
    n.iter <<- n.iter + 1
  }
  # increment rank counter
  rank <<- rank + 1
  # return the data 
  train.data
})
# clear junk
gc()
# report completion
print("done validating all spans for all ranks")

# bind train back together
train = bind_rows(train.split)

# clean memory
rm(n.iter, rank, train.split)

# ******************** #
# Full Dataset Models ====
# ******************** #

# ************************ #
# GAM: Full Dataset ====
# ************************ #
print("commence training gams on full data set")
gam.grid.full = expand.grid(ks = 5:30, 
                            fit.time = 0, 
                            predict.time = 0, 
                            train.rmse = 0, 
                            test.rmse = 0) %>%
  arrange(ks)

ks = 5:30

counter = 1
for(cp in ks){

  ################## 
  ## Fitting
  fit.time = now()
  fit = mgcv::bam(residual ~ m.rank + s(year, k = cp, by = m.rank),
                  data = train, knots = knts.era, method = "REML")
  gam.grid.full$fit.time[counter] = difftime(now(), fit.time, units = "secs")[[1]] %>% round(2)
  rm(fit.time)
  
  
  ################## 
  ## Predicting
  pred.time = now()
  # train predictions
  train.preds = predict(fit, train)
  test.preds  = predict(fit, test)
  gam.grid.full$predict.time[counter] = difftime(now(), pred.time, units = "secs")[[1]] %>% round(2)
  rm(pred.time)
  ###################
  ## Add Predictions
  train[[glue::glue("full.gam{cp}")]] = train.preds
  ## Test
  if(length(test.preds) == nrow(test)){
        empty.dt  = data.table(matrix(nrow = nrow(test)))
        empty.dt[[glue::glue("full.gam{cp}")]] = test.preds
        empty.dt = empty.dt[,2]
        temp = list(test, empty.dt)
        test <<- setDT(unlist(temp, recursive = F), 
                       check.names = T)
        rm(empty.dt, temp)
        }
  if(length(test.preds) != nrow(test)){
        print("incompatible vector length")
  }
  
  # calculate and add rmse
  gam.grid.full$train.rmse[counter] = get.rmse(train$residual, train.preds)
  gam.grid.full$test.rmse[counter]  = get.rmse(test$residual, test.preds)

  rm(train.preds, test.preds, fit)
  gc()
  
  counter = counter+1
  
}
# inform completion & clear junk
print("done validating all K's on full dataset")
gc()

# ****************************
# LOESS: Full Data Set ====
# ****************************
spans = c(.10,.15,.20,.25,.30,.40)
loess.grid.full = expand.grid(span = c(.10,.15,.20,.25,.30,.40),
                           fit.time = 0,
                           predict.time = 0,
                           train.rmse = 0,
                           test.rmse = 0) %>% 
  arrange(span)
start.loess = now()
for(i in 1:6){
  
  ##################
  ## Fitting
  fit.time = now()
  fit = locfit(residual ~ rank.m + lp(year, rank.m, nn = spans[i]),
                 data = train)
  loess.grid.full$fit.time[i] = difftime(now(), fit.time, units = "secs")[[1]] %>% round(2)
  rm(fit.time)
  ###################
  ## Predicting
  pred.time = now()
  train.preds = predict(fit, train)
  test.preds  = predict(fit, test)
  loess.grid.full$predict.time[i] = difftime(now(), pred.time, units = "secs")[[1]] %>% round(2)
  rm(pred.time)
  ###################
  ## Saving 
  train[[glue::glue("full.loe{spans[i]}")]] = train.preds
  ## save test preds if possible
  if(length(test.preds) == nrow(test)){
    empty.dt  = data.table(matrix(nrow = nrow(test)))
    empty.dt[[glue::glue("full.loe{spans[i]}")]] = test.preds
    empty.dt = empty.dt[,2]
    temp = list(test, empty.dt)
    test <<- setDT(unlist(temp, recursive = F), 
                   check.names = T)
    rm(temp, empty.dt)
        }
  if(length(test.preds) != nrow(test)){
    tst = length(test.preds)
    act = nrow(test)
    print("incompatible vector length")
    print(glue::glue("[1] pred.length = {tst} \n [1] act.length = {act}"))
    rm(tst, act)
    }
  ##################
  ## RMSE
  loess.grid.full$train.rmse[i]   = get.rmse(train$residual, 
                                               train.preds)
  loess.grid.full$test.rmse[i]    = get.rmse(test$residual,  
                                               test.preds)
  # clear junk
  rm(train.preds, test.preds, fit)
  gc()
}
print("done training and validating time effects") 
gc()



## filter out nas
holdout.na <- final_holdout_test %>% filter(is.na(userID))
final_holdout_test <- final_holdout_test %>% filter(!is.na(userID))
## split edx and holdout into ranks

edx.split = 
  edx %>%
  group_by(rank.m) %>%
  group_split()

holdout.split = 
  final_holdout_test %>%
  group_by(rank.m) %>%
  group_split()

# train the optimal gam and loess models on the split data
gam.models = lapply(edx.split, \(dframe){
  mgcv::bam(residual ~ s(year, k = 14), data = dframe,
            knots = knts.era, method = "REML")
})

loe.models = lapply(edx.split, \(dframe){
  locfit::locfit(residual ~ lp(year, nn = 0.1, deg = 3), data = dframe)
})

# after fitting make predictions on training and test
# gam
edx.split = map2(edx.split, gam.models, \(dframe, model){
  dframe$gam.rhat = predict(model, dframe)
  dframe
})
holdout.split = map2(holdout.split, gam.models, \(dframe, model){
  dframe$gam.rhat = predict(model, dframe)
  dframe
})
# loess
edx.split = map2(edx.split, loe.models, \(dframe, model){
  dframe$loe.rhat = predict(model, dframe)
  dframe
})
holdout.split = map2(holdout.split, loe.models, \(dframe, model){
  dframe$loe.rhat = predict(model, dframe)
  dframe
})

# bind everything together
edx = bind_rows(edx.split)
final_holdout_test = bind_rows(holdout.split)


# create an ensemble and assess performance
edx <-
  edx %>%
  mutate(ensemble = (gam.rhat+loe.rhat)*0.5)

final_holdout_test <-
  final_holdout_test %>%
  mutate(ensemble = (gam.rhat+loe.rhat)*0.5)

rm(gam.models,loe.models)
invisible(gc())

save.image("capstone_workspace_final.RData")