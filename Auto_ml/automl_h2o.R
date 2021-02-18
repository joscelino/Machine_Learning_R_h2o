library(h2o)

# Inicia a conexao com o servidor h20
h2o::h2o.init(nthreads = -1,
              ip = "localhost",
              port = 54321,
              max_mem_size = '12g')

# Diretorio raiz
ROOT_DIR <- getwd()

# Diretorio do arquivo
ARQ_DIR <- 
  'Dados'

# Nome do arquivo
ARQ_NOME <- 'Churn_treino.csv'

# Path do arquivo a ser carregado
arquivo_path <- 
  file.path(ROOT_DIR,
            ARQ_DIR,
            ARQ_NOME
  )


imp <- read.csv(arquivo_path, sep = ";")

# Pega nome da variavel dependente que deve estar na ultima coluna
y <- colnames(imp[length(imp)])

# transformando Df no formato H2o
dados <- as.h2o(imp)

# Separando dados de treino e teste
dados <- h2o.splitFrame(data = dados, ratios = 0.7)

treino <- dados[[1]]
teste <- dados[[2]]

# Transformando a variavel categorica
treino[,y] <- as.factor(treino[,y])
teste[,y] <- as.factor(teste[,y])

# Criando o modelo
modelo <- h2o.automl(x = colnames(treino[1:(length(imp) - 1)]),
                    y = y, training_frame = treino, max_runtime_secs = 60)

# Ranking dos modelos
lb <- as.data.frame(modelo@leaderboard)

# Importando dados para previsao

# Nome do arquivo
ARQ_NOME2 <- 'Churn_prever.csv'

# Path do arquivo a ser carregado
arquivo_path2 <- 
  file.path(ROOT_DIR,
            ARQ_DIR,
            ARQ_NOME2
  )

imp <- read.csv(arquivo_path2, sep = ";")
imp <- as.h2o(imp)

previsao <- h2o.predict(modelo@leader, imp)
previsao <- as.data.frame(previsao)

# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)
