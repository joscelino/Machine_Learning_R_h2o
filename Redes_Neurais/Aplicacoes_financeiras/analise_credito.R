library(caTools)
library(h2o)

# Importacao dos dados
dados <- read.csv("D:/Projetos_em_R/Machine_Learning/Dados/credit_data.csv")
dados$clientid <- NULL
view(dados)

# Divisao dos dados
divisao <- caTools::sample.split(dados$default, SplitRatio = 0.75)
base_treinamento <- base::subset(dados, divisao == TRUE)
base_teste <- base::subset(dados, divisao == FALSE)

# Acionando o pacote h20
h2o::h2o.init(nthreads = -1)

# Criando a rede neural
numeroNeuronios <- 100
epocas <- 1000
redeNeural <- h2o::h2o.deeplearning(y = 'default',
                                    training_frame = as.h2o(base_treinamento),
                                    activation = 'Rectifier',
                                    hidden = c(numeroNeuronios),
                                    epochs = epocas)

previsoes <- h2o::h2o.predict(redeNeural, 
                              newdata = as.h2o(base_teste[,-4]))
previsoes <- (previsoes > 0.5)
previsoes <- as.vector(previsoes)

matriz_confusao = table(base_teste[, 4], previsoes)
