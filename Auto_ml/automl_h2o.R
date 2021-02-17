library(h2o)
h2o.init()

imp = read.csv("D:/Projetos_em_R/Machine_Learning/Dados/Churn_treino.csv", sep = ";")

# Pega nome da variavel dependente que deve estar na ultima coluna
y = colnames(imp[length(imp)])

# transformando Df no formato H2o
dados = as.h2o(imp)

# Separando dados de treino e teste
dados = h2o.splitFrame(data = dados, ratios = 0.7)

treino = dados[[1]]
teste = dados[[2]]

# Transformando a variavel categorica
treino[,y] = as.factor(treino[,y])
teste[,y] = as.factor(teste[,y])

# Criando o modelo
modelo = h2o.automl(x = colnames(treino[1:(length(imp) - 1)]),
                    y = y, training_frame = treino, max_runtime_secs = 60)

# Ranking dos modelos
lb = as.data.frame(modelo@leaderboard)

# Importando dados para previsao
imp = read.csv("D:/Projetos_em_R/Machine_Learning/Dados/Churn_prever.csv", sep = ";")
imp = as.h2o(imp)

previsao = h2o.predict(modelo@leader, imp)
previsao = as.data.frame(previsao)

