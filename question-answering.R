library(tidyverse) #data manipulation
library(readxl) #load excel files
library(janitor)
library(splitstackshape)
library(text2vec) #NLP functions
library(readtext)
library(tm) 
library(stringr) #text manipulation
library(slam)
library(readxl)
library(Matrix)
library(reshape2)
library(koRpus)
library(superml)
library(udpipe)
library(textstem)
library(glmnet)
print('starting process...')
setwd("C:/Users/VP747QP/Desktop/AI Challenge")

#Load Data
train <- read.csv("train.csv",colClasses = c("character","character"))
paragraphs <- read.csv("paragraphs.csv") 
questions <- read.csv("questions.csv")
test <- read.csv("test.csv")


#Basic data processing/renaming steps
names(train) <- c("query","paragraph")
paragraphs <- paragraphs %>% clean_names()
questions <- questions %>% clean_names()
paragraphs$para_text <- as.character(paragraphs$para_text)
paragraphs$section_title <- as.character(paragraphs$section_title)
paragraphs$subsection_title <- as.character(paragraphs$subsection_title)
questions$qtext <- as.character(questions$qtext)

#function to clean all text - turn lowercase, remove numeric...
prep_fun = function(x) {
  x %>% 
    # make text lower case
    str_to_lower %>% 
    # remove non-alphanumeric symbols
    str_replace_all("[^[:alnum:]]", " ") %>% 
    # collapse multiple spaces
    str_replace_all("\\s+", " ")
}

#apply function above
paragraphs$para_text_clean <- prep_fun(paragraphs$para_text)
questions$qtext_clean <- prep_fun(questions$qtext)

#from glancing at questions, there were run-on/combined works
#I manually separated them out here but it's not a very efficient way to do this
questions$qtext_clean <- gsub("commissionsalary", "comission salary", questions$qtext_clean)
questions$qtext_clean <- gsub("taxexempt", "tax exempt", questions$qtext_clean)
questions$qtext_clean <- gsub("bonusaward", "bonus award", questions$qtext_clean)
questions$qtext_clean <- gsub("selfemployed", "self employed", questions$qtext_clean)
questions$qtext_clean <- gsub("taxfree", "tax free", questions$qtext_clean)
questions$qtext_clean <- gsub("owneremployee", "owner employee", questions$qtext_clean)
questions$qtext_clean <- gsub("taxsheltered", "tax sheltered", questions$qtext_clean)
questions$qtext_clean <- gsub("longterm", "long term", questions$qtext_clean)
questions$qtext_clean <- gsub("original issued discount", "oid", questions$qtext_clean)
questions$qtext_clean <- gsub("cashmethod", "cash method", questions$qtext_clean)

#Create combined dataframe with queries and paragraphs for vocabulary
quest <- data.frame(questions$qtext_clean)
para <- data.frame(paragraphs$para_text_clean)
names(quest) <- "text"
names(para) <- "text"
combined <- rbind(quest,para)
combined$text <- as.character(combined$text)

#Tokenize both queries and paragraphs while stemming the vocab
#token_query <- itoken(questions$qtext_clean,stem_tokenizer =function(x) {
#     lapply(word_tokenizer(x), SnowballC::wordStem, language="en")
#   },progressbar = FALSE)
#token_para<- itoken(paragraphs$para_text_clean,stem_tokenizer =function(x) {
#  lapply(word_tokenizer(x), SnowballC::wordStem, language="en")
#},progressbar = FALSE)

#function to tokenize words for processing using text2vec, stems words as well
stem_tokenizer1 =function(x) {
  tokens = word_tokenizer(x)
  lapply(tokens, SnowballC::wordStem, language="en")
}

#text2vec's token function
tokenized_text = itoken(combined$text,stem_tokenizer = stem_tokenizer1(),progressbar = FALSE)

#word2vec creating vocabulary (effectively TDM (term document matrix))
#ngram defines how many terms to take per token, since it is 2L, it will be 2
#prune_vocab defines words must be in the documents once and they cannot 
#be in more than 80% of documents - this removes overly common words
v = create_vocabulary(tokenized_text,ngram = c(1L,2L)) %>% 
  prune_vocabulary(term_count_min = 1,doc_proportion_max = 0.8)

vectorizer = vocab_vectorizer(v)


combined$index <- seq.int(nrow(combined))
questions$query <- seq.int(nrow(questions))

#creating document term matrix
dtm = create_dtm(tokenized_text, vectorizer)

#initializing tf-idf weighting
tfidf = TfIdf$new()

#apply tf-idf weighting to the DTM
dtm_tfidf = fit_transform(dtm, tfidf)

#calculate the cosine similarity beteen values and convert to dataframe
d1_d2_tfidf_cos_sim = sim2(x = dtm_tfidf, method = "cosine", norm = "l2")
tfidf_sim_df <- as.data.frame(summary(d1_d2_tfidf_cos_sim))

#filters out comparisons of the same query/doc, removes 
tfidf_sim_df <- tfidf_sim_df %>%
  filter(i != j, i <= NROW(questions$qid), j > NROW(questions$qid))

#processing step to get the correct id's back onto the responses
paragraphs$index <- paragraphs$i_para_id + 1009

tfidf_sim_df <- tfidf_sim_df %>%
  left_join(paragraphs, by = c("j" = "index")) %>%
    select(-2,-5:-12)

names(tfidf_sim_df) <- c("query","score","document")

#returning the top 5 documents for each query
tfidf_sim_df_top5 <- tfidf_sim_df %>%
  left_join(questions, by = c("query")) %>%
    select(-5:-6) %>%
      arrange(qid,desc(score)) %>%
        group_by(qid) %>%
          slice(1:5)

#return the top 1 value per query - used for assessing the accuracy
tfidf_sim_df_top1 <- tfidf_sim_df %>%
  left_join(questions, by = c("query")) %>%
  select(-5:-6) %>%
  arrange(qid,desc(score)) %>%
  group_by(qid) %>%
  slice(1)

tfidf_sim_df_top1$qid <- as.character(tfidf_sim_df_top1$qid)

evaluation <- train %>%
  left_join(tfidf_sim_df_top1, by = c("query" = "qid")) 

evaluation$match <- evaluation$paragraph == evaluation$document
evaluation <- evaluation %>%
  filter(!is.na(match))
paste0(sum(evaluation$match) / nrow(evaluation)*100,"% of the paragraph-query ID's match")

tfidf_sim_df_top5$qpid <- paste0(tfidf_sim_df_top5$qid,"#",tfidf_sim_df_top5$document)

test$qpid <- as.character(test$qpid)
test_alt <- test
test <- test %>%
  left_join(tfidf_sim_df_top5, by = c("qpid"))

test$score[!is.na(test$score)]<-1
test$score[is.na(test$score)]<-0

test <- test %>%
  select(-2,-3,-5,-6)

names(test) <- c("id","target") 

write.csv(test,"submissions.csv",row.names = FALSE)
print('Complete!')