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
X <- base::setdiff(names(layman_brothers.hex), c(Y,"ID"))

# Treino do Modelo
glm_model <-
  h2o::h2o.glm(training_frame = layman_brothers.train,
               x = X,
               y = Y,
               nfolds = 10,
               family = "binomial",
               alpha = 0.5,
               seed = seed,
               balance_classes = TRUE,
               stopping_metric = "MSE")

# Informacoes sobre o modelo
summary(glm_model)

# Predicoes
pred <-
  h2o::h2o.predict(object = glm_model,
                   newdata = layman_brothers.test)

# Sumario da classe 1 (DEFAULT)
summary(pred$p1)

# Matriz de confusao
h2o::h2o.confusionMatrix(glm_model)

# Importancia das variaveis
h2o::h2o.varimp_plot(glm_model)

# Desempenho do Modelo (Curva ROC)
perf <- 
  h2o::h2o.performance(glm_model,
                       layman_brothers.test)

plot(perf, type = "roc")

# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)

# Grafico dos Coeficientes com as magnitudes padronizadas
h2o.std_coef_plot(glm_model)

# Informacooes dos Coeficientes com as magnitudes padronizadas
glm_model@model$standardized_coefficient_magnitudes

# Perda do modelo 
h2o.logloss(perf)

# Informacao do AUC
h2o.auc(perf)


###############################################################################
############### Salvando o modelo para colocar em producao ####################                   
###############################################################################

#########
# Binario
#########
# Diretorio raiz
ROOT_DIR <- getwd()

# Diretorio do projeto
PROJECT_DIR <- 
  'glm'

artifact_path <- 
  file.path(ROOT_DIR,
            PROJECT_DIR
  )

model_path_object <- 
  h2o.saveModel(object=glm_model,
                path=artifact_path,
                force=TRUE
  )

print(model_path_object)

# Carrega o modelo na memoria (Binario)
saved_model <- h2o.loadModel(model_path_object)

# Predicao com o modelo recarregado em memoria
# com origem no arquivo binario
model_predict <- as.data.frame(
  h2o.predict(object = saved_model,
              newdata = layman_brothers.test,
  )
)

print(model_predict)

##############
# MOJO & POJO
##############
modelfile <- 
  h2o.download_mojo(glm_model,
                    path=artifact_path,
                    get_genmodel_jar=TRUE)

model_jar_path <- 
  paste(artifact_path, '/' ,modelfile, sep = "")


imported_model <- 
  h2o.import_mojo(mojo_file_path = model_jar_path)


# Predicao com o modelo recarregado em memoria
# com origem no arquivo MOJO
model_predict_imported <- as.data.frame(
  h2o.predict(object = imported_model,
              newdata = layman_brothers.test,
  )
)

print(model_predict_imported)

# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)

# Referencias
# [1] - http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html