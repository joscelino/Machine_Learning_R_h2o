library(h2o)
library(tibble)

# Inicia a conexao com o servidor h20
h2o::h2o.init(nthreads = -1,
              ip = "localhost",
              port = 54321,
              max_mem_size = '12g')

# Semente randomica global
seed <- 42

# Pega a URL do Layman Brothers no Github
layman_brothers_url <- 
  "https://raw.githubusercontent.com/fclesio/learning-space/master/Datasets/02%20-%20Classification/default_credit_card.csv"

# Minerando os dados
layman_brothers.hex <-
  h2o::h2o.importFile(path = layman_brothers_url,
                      destination_frame = "layman_brothers.hex")

# Transformando a variavel dependente como fator
layman_brothers.hex$DEFAULT <- 
  as.factor(layman_brothers.hex$DEFAULT)

# Informacoes sobre base de dados
base::summary(layman_brothers.hex)

# Visualizacao dos quantiles
stats::quantile(x = layman_brothers.hex$AGE, na.rm = TRUE)
h2o.hist(layman_brothers.hex$AGE)

# Agrupando os dados de educacao
education_stats <- h2o::h2o.group_by(data = layman_brothers.hex,
                                     by = "EDUCATION",
                                     nrow("EDUCATION"),
                                     gb.control = list(na.methods = "rm"))
education_stats

# Particionamento dos dados
layman_brothers.split <- 
  h2o::h2o.splitFrame(data = layman_brothers.hex,
                      ratios = 0.90, # Entrada pode ser vetor
                      seed = seed)


# Lista quantidade de dados separados
base::sapply(layman_brothers.split, function(x) nrow(x))

layman_brothers.train <- layman_brothers.split[[1]]
layman_brothers.test <- layman_brothers.split[[2]]

# Variavel dependente
Y <- "DEFAULT"

# Variaveis independentes
X <- base::setdiff(names(layman_brothers.hex), Y)




# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)