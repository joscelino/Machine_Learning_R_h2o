library(h2o)
library(h2oEnsemble)
library(readr)
library(tibble)

# Inicia a conexao com o servidor h20
h2o::h2o.init(nthreads = -1,
              max_mem_size = '12g')

# Mineracao dos dados
dados <- read.csv("D:/Projetos_em_R/Machine_Learning/Dados/echocardiogram.csv")
tibble::glimpse(dados)

dados[,"age"] <- base::as.numeric(dados[,"age"])
dados[,"alive"] <- base::as.factor(dados[,"alive"])

#Convertendo para dataframe h2o
dados <- h2o::as.h2o(dados)

# Breve estatistica
h2o::h2o.describe(dados)

# Manipulacao dos dados para construcao do modelo
Y <- "alive"
X <- base::setdiff(names(dados), Y)

dados[,Y] <- h2o::h2o.relevel(x = dados[,Y], y = "1")
h2o::h2o.levels(dados[,Y])

# Particonamento da base
ind <- h2o::h2o.splitFrame(dados, ratios = c(0.75, 0.1))

# Lista quantidade de dados separados
base::sapply(ind, function(x) nrow(x))

treino.h2o <- ind[[1]]
validacao.h2o <- ind[[2]]
teste.h2o <- ind[[3]]


################ Modelo Naive Bayes #########################################
nb <- h2o::h2o.naiveBayes(x = X, y = Y,
                          training_frame = treino.h2o,
                          validation_frame = validacao.h2o,
                          nfolds = 5,
                          fold_assignment = "Modulo",
                          keep_cross_validation_predictions = TRUE)

plot(h2o::h2o.performance(nb), type = "roc", pch = 16)
#############################################################################

################ Modelo Regressao Logistica #################################
rl <- h2o::h2o.glm(x = X, 
                   y = Y,
                   training_frame = treino.h2o,
                   validation_frame = validacao.h2o,
                   family = "binomial",
                   nfolds = 5,
                   fold_assignment = "Modulo",
                   keep_cross_validation_predictions = TRUE)

plot(h2o::h2o.performance(rl), type = "roc", pch = 16)
#############################################################################

################ Modelo Random Foret ########################################
rf <- h2o::h2o.randomForest(x = X, 
                            y = Y,
                            training_frame = treino.h2o,
                            validation_frame = validacao.h2o,
                            ntrees = 150,
                            mtries = 5,
                            max_depth = 3,
                            balance_classes = TRUE,
                            min_rows = 10,
                            stopping_metric = "AUC",
                            sample_rate = 1,
                            #binomial_double_trees = TRUE,
                            nfolds = 5,
                            fold_assignment = "Modulo",
                            keep_cross_validation_predictions = TRUE)

plot(h2o::h2o.performance(rf), type = "roc", pch = 16)
#############################################################################

################ Modelo Deep Learning #######################################
dl <- h2o::h2o.deeplearning(x = X, 
                            y = Y,
                            training_frame = treino.h2o,
                            validation_frame = validacao.h2o,
                            nfolds = 5,
                            fold_assignment = "Modulo",
                            keep_cross_validation_predictions = TRUE)

plot(h2o::h2o.performance(dl), type = "roc", pch = 16)
#############################################################################

################ Modelo Metaclassificador ###################################
# Lista com modelos treinados
modelos <- list(rf, nb, rl, dl)

# Definicao do metaclassificador
metalearner <- "h2o.gbm.wrapper"

# Modelo
super <- h2oEnsemble::h2o.stack(models = modelos,
                                response_frame = treino.h2o[,"alive"],
                                metalearner = metalearner,
                                seed = 1,
                                keep_levelone_data = TRUE)

utils::View(as.matrix(super$levelone))

plot(h2o::h2o.performance(super$metafit), type = "roc", pch = 16)
#############################################################################

pred <- h2oEnsemble::h2o.ensemble_performance(super, newdata = validacao.h2o)

print(pred, metric = 'AUC')

# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)
