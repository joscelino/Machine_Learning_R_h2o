library(h2o)

# Inicia a conexao com o servidor h20
h2o::h2o.init(nthreads = -1,
              ip = "localhost",
              port = 54321,
              max_mem_size = '12g')

# Semente randomica global
seed <- 42

# Pega a URL do Residencias no Github
residential_url <- 
  "https://raw.githubusercontent.com/fclesio/learning-space/master/Datasets/04%20-%20Linear%20Regression/residential_building_dataset.csv"

# Minerando os dados
residential.hex <- 
  h2o.importFile(path = residential_url,
                 destination_frame = "residential.hex")

# Garantindo variavel dependente como numerica
residential.hex$out_v1 <- as.numeric(residential.hex$out_v1)
residential.hex$out_v2 <- as.numeric(residential.hex$out_v2)

# Informacoes sobre base de dados
base::summary(residential.hex)

# Visualizacao dos quantiles
stats::quantile(x = layman_brothers.hex$AGE, na.rm = TRUE)
h2o.hist(layman_brothers.hex$AGE)

# Agrupando os dados de Financiamento
physical_financial_v1_stats <- 
  h2o::h2o.group_by(data = residential.hex,
                    by = "physical_financial_v1",
                    nrow("physical_financial_v1"),
                    gb.control = list(na.methods = "rm"))

physical_financial_v1_stats

# Particionamento dos dados
residential_building.split<-
  h2o.splitFrame(data = residential.hex,
                 ratios = 0.90,
                 seed = seed)

# Lista quantidade de dados separados
base::sapply(residential_building.split, function(x) nrow(x))

residential_building.train <- residential_building.split[[1]]
residential_building.test <- residential_building.split[[2]]


# Variavel dependente
Y <- "out_v1"

# Variaveis independentes
X <- base::setdiff(names(residential.hex), Y)




# Encerra conexao
h2o::h2o.shutdown(prompt = FALSE)
