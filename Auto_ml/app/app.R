library(shiny)
library(shinybusy)
library(h2o)

h2o.init()


ui <- fluidPage(
    titlePanel("Auto Machine Learning"),
    tabsetPanel(
        tabPanel("Criacao do Modelo",
            fluidRow(
                column(4, fileInput("arquivo", "Selecione o arquivo:", multiple=F, accept=c(".csv"))),
                column(4, numericInput("Tempo", "Tempo maximo (min.) :", value=1, min=1)),
                column(4, actionButton("Processar", "Processar"))
            ),
            fluidRow(
                column(4, tableOutput("Dados"))
                
            )
        ),
        tabPanel("Previsao",
            fluidRow(
                column(6, fileInput("arquivo2", "Selecione o arquivo:", multiple=F, accept=c(".csv"))),
                column(6, actionButton("Processar2", "Processar"))
            ),
            fluidRow(
                column(4, tableOutput("prev"))
            )
        )
    )
)


server <- function(input, output) {
    
    observeEvent(input$Processar, {
        
        file1 = input$arquivo
        
        imp = read.csv(file1$datapath, sep = ";")

        y = colnames(imp[length(imp)])
        
        dados = as.h2o(imp)
        
        dados = h2o.splitFrame(data = dados, ratios = 0.7)
        
        treino = dados[[1]]
        teste = dados[[2]]
        
        treino[,y] = as.factor(treino[,y])
        teste[,y] = as.factor(teste[,y])
        
        show_modal_spinner()
        modelo <<- h2o.automl(x = colnames(treino[1:(length(imp) - 1)]),
                            y = y, training_frame = treino, max_runtime_secs = input$Tempo * 60)
        remove_modal_spinner()
        
        lb = as.data.frame(modelo@leaderboard)
        
        output$Dados = renderTable({lb})
        

    })
    
    observeEvent(input$Processar2, {
        
        file2 = input$arquivo2
        
        imp = read.csv(file2$datapath, sep = ";")
        imp = as.h2o(imp)
        
        previsao = h2o.predict(modelo@leader, imp)
        previsao = as.data.frame(previsao)
        
        output$prev = renderTable({previsao})
    
    })

}

shinyApp(ui = ui, server = server)
