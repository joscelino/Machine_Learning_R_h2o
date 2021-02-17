library(h2o)

# Semente randomica global
seed <- 42

?h2o.init

# Inicia a conexao com o servidor h20
h2o::h2o.init(nthreads = -1,
              max_mem_size = '12g')

?h2o::h2o.glm

demo(h2o::h2o.glm)

# Pega a URL do Layman Brothers no Github
layman_brothers_url <- 
  "https://raw.githubusercontent.com/fclesio/learning-space/master/Datasets/02%20-%20Classification/default_credit_card.csv"

# Minerando os dados
layman_brothers.hex <-
  h2o::h2o.importFile(path = layman_brothers_url,
                      destination_frame = "layman_brothers.hex")


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

# Particonamento dos dados
layman_brothers.split <- 
  h2o::h2o.splitFrame(data = layman_brothers.hex,
                      ratios = 0.75, # Entrada pode ser vetor
                      seed = seed)
# Lista quantidade de dados separados
base::sapply(layman_brothers.split, function(x) nrow(x))

layman_brothers.train <- layman_brothers.split[[1]]
layman_brothers.test <- layman_brothers.split[[2]]

# Variavel dependente
Y <- "DEFAULT"

# Variaveis independentes
X <- base::setdiff(names(layman_brothers.hex), Y)

# Modelo GLM simples
layman_brothers.glm <- 
  h2o::h2o.glm(training_frame = layman_brothers.train,
               x = X,
               y = Y,
               family = "binomial",
               alpha = 0.5,
               seed = seed)

# Informacoes do modelo
summary(layman_brothers.glm)

# Realizando predicoes com o modelo
pred <- 
  h2o::h2o.predict(object = layman_brothers.glm,
                   newdata = layman_brothers.test)

# Informacoes de predicao em relacao a sua probabilidade 
summary(pred$p1)

# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)

# Matriz de confusao
h2o::h2o.confusionMatrix(layman_brothers.glm)

# Importancia das variaveis
h2o::h2o.varimp_plot(layman_brothers.glm)

# Curva ROC do modelo
perf <- h2o::h2o.performance(layman_brothers.glm,
                             layman_brothers.test)
plot(perf, type = "roc")

# Garfico de coeficientes com magnitudes padrao
h2o::h2o.std_coef_plot(layman_brothers.glm)

# Desempenho do modelo
perf

# Perda do modelo AUC (Grau de convergencia do modelo - quanto menor, melhor)
h2o::h2o.auc(perf)

# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)
