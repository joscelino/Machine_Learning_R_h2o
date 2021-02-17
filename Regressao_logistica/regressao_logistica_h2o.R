library(h2o)

# Inicia a conexao com o servidor h20
h2o::h2o.init(nthreads = -1,
              max_mem_size = '12g')

# Mineracao dos dados
dados <- h2o::h2o.importFile(path = "D:/Projetos_em_R/Machine_Learning/Dados/Dataset_Classificacao_Contratacao_TP.csv",
                             header = TRUE, sep = ";")

# Breve estatistica
h2o::h2o.describe(dados)

# Particionamento dos dados
split_dados <- h2o::h2o.splitFrame(dados, ratios = c(0.8, 0.1))

# Verificando os tamanhos dos particonamentos
base::sapply(split_dados, function(x) nrow(x))

treino.h2o <- split_dados[[1]]
validacao.h2o <- split_dados[[2]]
teste.h2o <- split_dados[[3]]

# Liberando memoria para o h2o
rm(split_dados)

# Variaveis projetoras
X <- c('Prova_Logica',
       'Redacao',
       'Psicotecnico',
       'Dinamica_Grupo',
       'Fit_Cultural',
       'Ingles',
       'Avaliacao_RH',
       'Auto_Avaliacao',
       'Demograficos',
       'Estado_Civil',
       'Escolaridade')
Y <- 'Classe'

# Modelo
regressao_h2o <- h2o::h2o.glm(x = X,
                               y = Y,
                               training_frame = treino.h2o,
                               validation_frame = validacao.h2o,
                               family = 'binomial',
                               link = 'logit',
                               alpha = 1, # regressao lasso
                               lambda_search = TRUE,
                               balance_classes = FALSE, # Setar TRUE para qtde diferentes entre classes
                               remove_collinear_columns = TRUE,
                               missing_values_handling = "MeanImputation", # Tratar NAs
                               standardize = TRUE, # Normalizar os dados
                               ignore_const_cols = TRUE,
                               nfolds = 6)
# Verificando os dados de treinamento
regressao_h2o

# Plotando graficos
# Curva ROC
plot(h2o::h2o.performance(regressao_h2o), type = 'roc')

# Variaveis mais importantes
h2o::h2o.varimp_plot(regressao_h2o)

# Matriz de confusao
h2o::h2o.confusionMatrix(object = regressao_h2o,
                         newdata = teste.h2o)

# Salvando o modelo para trasferencia de aprendizado
h2o::h2o.saveModel(object = regressao_h2o, path = getwd(), force = TRUE)

# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)
