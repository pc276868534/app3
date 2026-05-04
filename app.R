# --- 1. 依赖包加载 ---
library(shiny)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3extralearners)
library(ggplot2)
library(shapviz)
library(kernelshap)
library(data.table)

# --- 2. 辅助函数与配置 ---
display_names_map <- c(
  "AST" = "AST (U/L)",
  "PLT" = "PLT (×10^9/L)", 
  "gender" = "Gender",
  "number.of.metastatic.organs" = "Number of metastatic organs (n)",
  "other.site.metastasis" = "Other site metastasis (n)",
  "primary.tumor.sites" = "Primary tumor site",
  "grade" = "Grade"
)

gender_choices <- list("Male" = 0, "Female" = 1)
site_choices   <- list("left colon cancer" = 1, "right colon cancer" = 2, "rectal cancer" = 3)
grade_choices  <- list(
  "moderately to well differentiated adenocarcinoma" = 1,
  "poorly differentiated adenocarcinoma" = 2,
  "adenocarcinoma, unknown type" = 3
)

clean_name_for_plot <- function(x) {
  name <- if (x %in% names(display_names_map)) display_names_map[[x]] else gsub("\\.", " ", x)
  return(gsub("\\s*\\(.*?\\)", "", name)) 
}

# --- 3. 加载模型逻辑 (增加容错) ---
model_file <- "image_ChooseModel.RData"

if (file.exists(model_file)) {
  load(model_file)
  # 确保加载的对象存在
  if(!exists("graph_pipeline_ChooseModel")) stop("Object 'graph_pipeline_ChooseModel' not found in RData!")
  
  # 重新训练或更新状态 (根据mlr3习惯)
  graph_pipeline_ChooseModel$param_set$set_values(.values = best_ChooseModel_param_vals)
  graph_pipeline_ChooseModel$train(task_train)
  
  task_model <- model_ChooseModel_aftertune$state$train_task
  variables <- setNames(as.list(task_model$feature_types$type), task_model$feature_types$id)
  train_data <- as.data.frame(task_train$data())
  
  set.seed(42)
  # 选取代表性背景集
  bg_X_minimal <- train_data[sample(nrow(train_data), min(20, nrow(train_data))), task_model$feature_names, drop = FALSE]
} else {
  # 模拟数据仅用于调试UI，实际运行需确保文件存在
  stop("Critical Error: 'image_ChooseModel.RData' not found. Please upload the file.")
}

# --- 4. UI 界面 ---
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body { background-color: #f4f7f9; font-family: 'Segoe UI', sans-serif; }
      .navbar-custom { background-color: #2c77b4; color: white; padding: 15px; margin-bottom: 25px; border-radius: 0 0 10px 10px; }
      .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
      .section-title { color: #2c77b4; font-weight: bold; border-left: 4px solid #2c77b4; padding-left: 10px; margin-bottom: 15px; }
      .risk-container { position: relative; width: 100%; height: 20px; background: linear-gradient(to right, #5cb85c 30%, #f0ad4e 30% 50%, #d9534f 50%); border-radius: 10px; margin: 20px 0; }
      .risk-indicator { position: absolute; top: -5px; width: 4px; height: 30px; background-color: #333; transform: translateX(-50%); transition: left 0.5s ease; }
      .prob-text { font-size: 20px; font-weight: bold; color: #333; }
    "))
  ),
  
  div(class = "navbar-custom", h2("PM Risk Prediction Model for Colorectal Cancer")),
  
  fluidRow(
    column(width = 4,
           div(class = "card",
               div(class = "section-title", "Input Features"),
               # 动态生成输入框
               lapply(names(variables), function(feature) {
                 label_text <- if(feature %in% names(display_names_map)) display_names_map[[feature]] else gsub("\\.", " ", feature)
                 
                 if (feature == "gender") {
                   selectInput(feature, label_text, choices = gender_choices)
                 } else if (feature == "primary.tumor.sites") {
                   selectInput(feature, label_text, choices = site_choices)
                 } else if (feature == "grade") {
                   selectInput(feature, label_text, choices = grade_choices)
                 } else if (variables[[feature]] %in% c("numeric", "integer")) {
                   numericInput(feature, label_text, value = round(median(train_data[[feature]], na.rm = TRUE), 2))
                 } else {
                   selectInput(feature, label_text, choices = task_model$levels(feature)[[1]])
                 }
               }),
               actionButton("predict", "Calculate Risk", class = "btn-primary", style="width:100%; margin-top:10px;")
           )
    ),
    
    column(width = 8,
           div(class = "card",
               div(class = "section-title", "Prediction Results"),
               uiOutput("prob_text"),
               div(class = "risk-container", uiOutput("indicator")),
               uiOutput("risk_badge")
           ),
           
           div(class = "card",
               div(class = "section-title", "SHAP Waterfall Interpretation"),
               plotOutput("waterfall", height = "500px")
           )
    )
  )
)

# --- 5. Server 逻辑 ---
server <- function(input, output, session) {
  
  # 封装预测逻辑
  prediction_data <- eventReactive(input$predict, {
    # 构造输入行
    input_list <- lapply(names(variables), function(f) {
      val <- input[[f]]
      if (variables[[f]] %in% c("numeric", "integer")) return(as.numeric(val))
      return(val)
    })
    input_df <- as.data.frame(input_list)
    colnames(input_df) <- names(variables)
    
    # 因子类型严格转换
    for (f in names(variables)) {
      if (variables[[f]] == "factor") {
        input_df[[f]] <- factor(input_df[[f]], levels = task_model$levels(f)[[1]])
      }
    }
    
    # 预测
    tryCatch({
      pred <- model_ChooseModel_aftertune$predict_newdata(input_df)
      prob <- round(as.numeric(as.data.table(pred)$prob.1), 3)
      list(df = input_df, prob = prob, error = NULL)
    }, error = function(e) {
      list(df = NULL, prob = 0, error = e$message)
    })
  }, ignoreNULL = FALSE)
  
  output$prob_text <- renderUI({
    res <- prediction_data()
    if(!is.null(res$error)) return(p(paste("Error:", res$error), style="color:red;"))
    div(class = "prob-text", paste("Predicted Probability:", res$prob))
  })
  
  output$indicator <- renderUI({
    res <- prediction_data()
    prob <- res$prob
    # 计算指示器位置
    pos <- if (prob <= 0.3) (prob / 0.3) * 30 
    else if (prob <= 0.5) 30 + ((prob - 0.3) / 0.2) * 20 
    else 50 + ((prob - 0.5) / 0.5) * 50
    tags$div(class = "risk-indicator", style = paste0("left: ", max(0, min(100, pos)), "%;"))
  })
  
  output$risk_badge <- renderUI({
    prob <- prediction_data()$prob
    if(prob > 0.5) {
      tags$span("High Risk", style="background:#d9534f; color:white; padding:5px 15px; border-radius:10px; font-weight:bold;")
    } else if(prob >= 0.3) {
      tags$span("Medium Risk", style="background:#f0ad4e; color:white; padding:5px 15px; border-radius:10px; font-weight:bold;")
    } else {
      tags$span("Low Risk", style="background:#5cb85c; color:white; padding:5px 15px; border-radius:10px; font-weight:bold;")
    }
  })
  
  output$waterfall <- renderPlot({
    res <- prediction_data()
    if(is.null(res$df)) return(NULL)
    
    # 定义预测包装函数
    p_fun <- function(obj, newdata) {
      as.numeric(as.data.table(obj$predict_newdata(newdata))$prob.1)
    }
    
    withProgress(message = 'Calculating SHAP values...', {
      # 计算 SHAP
      shap_vals <- kernelshap(
        model_ChooseModel_aftertune, 
        res$df, 
        bg_X = bg_X_minimal, 
        pred_fun = p_fun
      )
      
      # 清洗列名
      colnames(shap_vals$S) <- sapply(colnames(shap_vals$S), clean_name_for_plot)
      colnames(shap_vals$X) <- sapply(colnames(shap_vals$X), clean_name_for_plot)
      
      sv_waterfall(shapviz(shap_vals)) + 
        theme_minimal(base_size = 14) +
        labs(title = "Feature Contribution (SHAP)", x = "Probability Impact")
    })
  })
}

shinyApp(ui, server)
