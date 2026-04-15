try(library(shiny))
try(library(mlr3))
try(library(mlr3learners))
try(library(mlr3pipelines))
try(library(mlr3extralearners))
try(library(ggplot2))
try(library(shapviz))
try(library(kernelshap))

# 定义特征信息
feature_config <- list(
  "hepatic.metastasis" = list(
    name = "hepatic metastasis 0/1",
    type = "factor",
    choices = list("No" = 0, "Yes" = 1),
    default = 0
  ),
  "peritoneal.metastasis" = list(
    name = "peritoneal metastasis 0/1",
    type = "factor",
    choices = list("No" = 0, "Yes" = 1),
    default = 0
  ),
  "neutrophil" = list(
    name = "neutrophil (x10^9/L)",
    type = "numeric",
    min = 0,
    max = 20,
    step = 0.1,
    default = 4.29
  ),
  "PLT" = list(
    name = "PLT (x10^9/L)",
    type = "numeric",
    min = 0,
    max = 1000,
    step = 1,
    default = 290
  ),
  "LDH" = list(
    name = "LDH U/L",
    type = "numeric",
    min = 0,
    max = 2000,
    step = 1,
    default = 200
  )
)

# 特征显示顺序
feature_order <- names(feature_config)

# 1. 加载模型
try(load("image_ChooseModel.RData"))
try(graph_pipeline_ChooseModel$param_set$set_values(.values = best_ChooseModel_param_vals))
try(graph_pipeline_ChooseModel$train(task_train))

# 2. 获取模型信息
task_model <- model_ChooseModel_aftertune$state$train_task
train_data <- as.data.frame(task_train$data())

# 3. 动态生成 Shiny UI
ui <- fluidPage(
  titlePanel("Lung Metastasis Prediction Model for Colorectal Cancer Patients"),
  sidebarLayout(
    sidebarPanel(
      width = 4,
      h4("Input Features", style = "color: #2c77b4; font-weight: bold;"),
      
      # 生成5个特征的输入组件
      lapply(feature_order, function(feature) {
        config <- feature_config[[feature]]
        
        if (config$type == "factor") {
          input_control <- selectInput(
            inputId = feature, 
            label = config$name, 
            choices = config$choices,
            selected = as.character(config$default)
          )
        } else if (config$type == "numeric") {
          input_control <- numericInput(
            inputId = feature, 
            label = config$name, 
            value = config$default,
            min = config$min,
            max = config$max,
            step = config$step
          )
        }
        
        div(style = "margin-bottom: 15px;", input_control)
      }),
      
      div(style = "text-align: center; margin-top: 20px;",
          actionButton("predict", "Predict Now", 
                       class = "btn-primary", 
                       style = "width: 80%; height: 40px; font-size: 16px;")
      )
    ),
    
    mainPanel(
      width = 8,
      
      # 预测结果部分
      h3("Prediction Result", style = "color: #2c77b4;"),
      div(style = "background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px;",
          uiOutput("prediction_result"),
          uiOutput("risk_indicator_ui")
      ),
      
      # 全局SHAP图表
      h3("Global SHAP Analysis", style = "color: #2c77b4; margin-top: 30px;"),
      plotOutput("global_shap_importance", width = "100%", height = "300px"),
      plotOutput("global_shap_beeswarm", width = "100%", height = "300px"),
      
      # 个体SHAP图表
      h3("Individual SHAP Analysis for Current Input", style = "color: #2c77b4; margin-top: 30px;"),
      plotOutput("individual_waterfall", width = "100%", height = "350px"),
      plotOutput("individual_force", width = "100%", height = "300px")
    )
  )
)

# 4. Shiny Server 逻辑
server <- function(input, output) {
  
  # 渲染全局SHAP图表
  output$global_shap_importance <- renderPlot({
    if (exists("SHAP_sv_ChooseModel")) {
      sv_importance(SHAP_sv_ChooseModel) + 
        theme_minimal() +
        theme(axis.text = element_text(size = 12), 
              axis.title = element_text(size = 14),
              plot.title = element_text(size = 16, face = "bold"))
    } else {
      ggplot() + 
        annotate("text", x = 0.5, y = 0.5, label = "Global SHAP data not available", size = 6) +
        theme_void()
    }
  })
  
  output$global_shap_beeswarm <- renderPlot({
    if (exists("SHAP_sv_ChooseModel")) {
      sv_importance(SHAP_sv_ChooseModel, kind = "beeswarm") + 
        theme_minimal() +
        theme(axis.text = element_text(size = 12), 
              axis.title = element_text(size = 14),
              plot.title = element_text(size = 16, face = "bold"))
    } else {
      ggplot() + 
        annotate("text", x = 0.5, y = 0.5, label = "Global SHAP data not available", size = 6) +
        theme_void()
    }
  })
  
  observeEvent(input$predict, {
    # 收集用户输入
    input_data <- lapply(feature_order, function(feature) {
      config <- feature_config[[feature]]
      val <- input[[feature]]
      
      if (config$type == "numeric") {
        as.numeric(val)
      } else {
        as.numeric(val)  # 对于因子，获取数值
      }
    })
    
    input_df <- as.data.frame(input_data)
    names(input_df) <- feature_order
    print("Input data:")
    print(input_df)
    
    # 确保数据类型正确
    for (feature in feature_order) {
      config <- feature_config[[feature]]
      if (config$type == "factor") {
        # 转换因子
        input_df[[feature]] <- factor(input_df[[feature]], levels = c(0, 1), labels = c("No", "Yes"))
      } else if (config$type == "numeric") {
        input_df[[feature]] <- as.numeric(input_df[[feature]])
      }
    }
    
    # 进行预测
    prediction <- model_ChooseModel_aftertune$predict_newdata(input_df)
    prob <- round(as.numeric(as.data.table(prediction)$prob.1), 3)
    
    # 显示预测结果
    output$prediction_result <- renderUI({
      HTML(paste("<div style='font-size: 20px; color: #2c77b4; font-weight: bold; margin-bottom: 10px;'>",
                 "The probability that this patient has lung metastasis is: ",
                 "<span style='color: #d9534f;'>", prob, "</span>",
                 "</div>"))
    })
    
    # 更新风险指示器
    output$risk_indicator_ui <- renderUI({
      # 计算风险等级
      risk_level <- if(prob > 0.5) {
        list(color = "#d9534f", label = "High Risk", pos = 50 + ((prob - 0.5) / 0.5) * 50)
      } else if(prob >= 0.3) {
        list(color = "#f0ad4e", label = "Medium Risk", pos = 30 + ((prob - 0.3) / 0.2) * 20)
      } else {
        list(color = "#5cb85c", label = "Low Risk", pos = (prob / 0.3) * 30)
      }
      
      tagList(
        # 风险条
        div(style = "position: relative; width: 100%; height: 20px; background: linear-gradient(to right, #5cb85c 0%, #5cb85c 30%, #f0ad4e 30%, #f0ad4e 50%, #d9534f 50%, #d9534f 100%); border-radius: 10px; margin: 10px 0;",
            div(style = paste0("position: absolute; top: -2px; left: ", min(100, max(0, risk_level$pos)), "%; width: 4px; height: 24px; background: #000; border-radius: 2px; transform: translateX(-50%);"))
        ),
        # 风险标签
        div(style = "position: relative; width: 100%; height: 20px; margin-bottom: 10px;",
            span(style = "position: absolute; left: 0%;", "0 Low Risk"),
            span(style = "position: absolute; left: 30%;", "0.3"),
            span(style = "position: absolute; left: 50%;", "0.5"),
            span(style = "position: absolute; right: 0%;", "High Risk 1")
        ),
        # 风险徽章
        div(style = "text-align: center;",
            span(style = paste0("display: inline-block; padding: 5px 20px; background: ", risk_level$color, 
                                "; color: white; border-radius: 20px; font-weight: bold; font-size: 16px;"),
                 risk_level$label)
        )
      )
    })
    
    # 计算个体SHAP值
    compute_individual_shap(input_df, output, prob)
  })
  
  # 独立的SHAP计算函数
  compute_individual_shap <- function(input_df, output, prob) {
    # 获取背景数据
    background_data <- train_data[, feature_order, drop = FALSE]
    input_features <- input_df[, feature_order, drop = FALSE]
     
    individual_shap <- NULL
    shap_method_used <- "none"
     
    # 使用kernelshap
    if (is.null(individual_shap)) {
      tryCatch({
        print("Trying kernel SHAP method...")
        
        # 定义预测函数
        pred_fun <- function(object, newdata) {
          pred <- object$predict_newdata(newdata)
          if ("prob.1" %in% names(as.data.table(pred))) {
            as.numeric(as.data.table(pred)$prob.1)
          } else {
            as.numeric(as.data.table(pred)$response)
          }
        }
        
        # 使用kernelshap计算
        shap_values <- kernelshap(
          object = model_ChooseModel_aftertune,
          X = input_features,
          bg_X = background_data[sample(nrow(background_data), min(50, nrow(background_data))), ],
          pred_fun = pred_fun
        )
        
        # 使用SHAP图表专用名称
        colnames(shap_values$S) <- sapply(feature_order, function(f) feature_config[[f]]$name)
        colnames(shap_values$X) <- sapply(feature_order, function(f) feature_config[[f]]$name)
        
        individual_shap <- shapviz(shap_values)
        shap_method_used <- "kernelshap"
        print("Kernel SHAP successful!")
        
      }, error = function(e) {
        print(paste("Kernel SHAP failed:", e$message))
      })
    }
     
    # 如果成功计算了SHAP值
    if (!is.null(individual_shap)) {
      print(paste("SHAP calculation successful using method:", shap_method_used))
      
      # 自定义主题
      theme_custom <- function() {
        theme_minimal() + 
          theme(
            panel.grid = element_blank(),
            panel.border = element_blank(),
            panel.background = element_blank(),
            plot.background = element_blank(),
            plot.title = element_text(hjust = 0.5, face = "bold", size = 16, 
                                      margin = margin(b = 10)),
            axis.line.x = element_line(color = "black", linewidth = 0.8),
            axis.line.y = element_line(color = "black", linewidth = 0.8),
            axis.text = element_text(size = 12, color = "black", face = "bold"),
            axis.title = element_text(size = 13, face = "bold"),
            axis.ticks = element_line(color = "black", linewidth = 0.5),
            axis.ticks.length = unit(2.5, "pt"),
            legend.position = "none"
          )
      }
      
      # 自定义颜色
      positive_color <- "#9C27B0"  # 紫红色
      negative_color <- "#FBC02D"  # 黄色
      
      # 渲染个体SHAP waterfall图
      output$individual_waterfall <- renderPlot({
        p <- sv_waterfall(individual_shap, row_id = 1) + 
          theme_custom() +
          labs(title = "SHAP Waterfall Plot for Current Input",
               x = "SHAP Value",
               y = "") +
          scale_fill_manual(values = c("FALSE" = negative_color, "TRUE" = positive_color)) +
          theme(
            axis.text.x = element_text(size = 12, face = "bold"),
            axis.text.y = element_text(size = 12, face = "bold", hjust = 1)
          )
        print(p)
      })
      
      # 渲染个体SHAP force图
      output$individual_force <- renderPlot({
        p <- sv_force(individual_shap, row_id = 1) + 
          theme_custom() +
          labs(title = "SHAP Force Plot for Current Input",
               x = "Prediction Value",
               y = "") +
          scale_color_gradient2(
            low = negative_color,  # 黄色
            mid = "#9E9E9E",       # 灰色
            high = positive_color, # 紫红色
            midpoint = median(as.data.frame(individual_shap$X)[1, ], na.rm = TRUE)
          ) +
          theme(
            axis.text.x = element_text(size = 12, face = "bold"),
            axis.text.y = element_blank(),
            axis.line.y = element_blank()
          )
        print(p)
      })
      
    } else {
      # 所有方法都失败了
      print("All SHAP methods failed")
      output$individual_waterfall <- renderPlot({
        ggplot() + 
          annotate("text", x = 0.5, y = 0.5, label = "SHAP calculation failed for this input", size = 6) +
          theme_void()
      })
      
      output$individual_force <- renderPlot({
        ggplot() + 
          annotate("text", x = 0.5, y = 0.5, label = "SHAP calculation failed for this input", size = 6) +
          theme_void()
      })
    }
  }
}

# 5. 运行 Shiny App
shinyApp(ui = ui, server = server)
