library(readxl)
library(h2o)

# Inicia a conexao com o servidor h20
h2o::h2o.init(nthreads = -1,
              max_mem_size = '12g')

# Mineracao dos dados
dados <- readxl::read_excel("D:/Projetos_em_R/Machine_Learning/Dados/Dataset_Classificacao_RiscoCredito.xlsx")
base::anyNA(dados)

dados$Estado_Civil <- base::factor(dados$Estado_Civil)
dados$Escolaridade <- base::factor(dados$Escolaridade)
dados$Metodo_Pagamento <- base::factor(dados$Metodo_Pagamento)

dados$Classe <- base::factor(dados$Classe,
                             levels = c('Alto_Risco', 'Baixo_Risco'))

#Convertendo para dataframe h2o
dados.h2o <- h2o::as.h2o(dados)

# Particionamento dos dados
split_dados <- h2o::h2o.splitFrame(dados.h2o, ratios = c(0.75, 0.1))

# Lista quantidade de dados separados
base::sapply(split_dados, function(x) nrow(x))

# Manipulando os dados
treino.h2o <- split_dados[[1]]
validacao.h2o <- split_dados[[2]]
teste.h2o <- split_dados[[3]]

# Definindo a varivale de resposta
Y <- 'Classe'

fit <- h2o::h2o.gbm(y = Y,
                    training_frame = treino.h2o,
                    validation_frame = validacao.h2o,
                    learn_rate = 0.08,
                    max_depth = 25,
                    col_sample_rate = 0.8,
                    min_rows = .75,
                    ntrees = 100,
                    nfolds = 5,
                    seed = 1,
                    balance_classes = TRUE)

# Perfomance do modelo
h2o::h2o.performance(fit, teste.h2o)

##### GRID SEARCH
# Definindo os hiperparametros
gbm_parametros <- base::list(learn_rate = c(0.01, 0.1),
                             max_depth = c(15, 30),
                             sample_rate = c(0.6, 1.0),
                             col_sample_rate = c(0.2, 0.7, 1.0),
                             ntrees = c(500, 800, 1000),
                             min_rows = c(10, 20),
                             categorical_encoding = c("OneHotExplicit",
                                                      "Eigen", "LabelEncoder"))

# Inicializa o grid seach
gbm_grid1 <- h2o::h2o.grid("gbm", y = Y,
                           training_frame = treino.h2o,
                           validation_frame = validacao.h2o,
                           seed = 1,
                           balance_classes = TRUE,
                           hyper_params = gbm_parametros,
                           max_runtime_secs = 3600)

utils::View(gbm_grid1@summary_table)


# Executando o melhor modelo
best_gbm_model_id <- h2o::h2o.getModel(gbm_grid1@model_ids[[1]])

h2o.performance(best_gbm_model_id, teste.h2o)

h2o.varimp_plot(best_gbm_model_id)
