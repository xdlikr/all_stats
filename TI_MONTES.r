#' 计算非稳定性指标的耐受区间 (No Slope Method)
#'
#' 该函数根据 Richard K. Burdick 等人的论文方法，为一个随机效应模型计算单侧和双侧耐受区间。
#' 适用于不存在时间趋势（即斜率为零）的数据。
#'
#' @param InputData 一个数据框 (data.frame)，必须包含以下三列：
#'   - Y: 数值型，表示属性的测量结果。
#'   - Lot: 因子或字符型，表示批次名称。
#'   - Months: 数值型，表示测量时所处的月龄 (在此函数中未使用，但为保持数据结构一致性而提及)。
#' @param ConfLevel 数值型，表示置信水平 (例如, 0.95)。
#' @param PropCover 数值型，表示覆盖总体比例 (例如, 0.99)。
#'
#' @return 返回一个包含 6 列的矩阵，分别为不同方法计算出的耐受区间下限 (L) 和上限 (U)。
#'
#' @export
#' @examples
#' # 示例用法 (需要创建一个示例数据框)
#' # set.seed(123)
#' # example_data <- data.frame(
#' #   Y = c(rnorm(5, 100, 2), rnorm(5, 102, 2), rnorm(5, 99, 2)),
#' #   Lot = factor(rep(c("A", "B", "C"), each = 5)),
#' #   Months = rep(1:5, 3)
#' # )
#' # NoSlope(InputData = example_data, ConfLevel = 0.95, PropCover = 0.99)

NoSlope <- function(InputData, ConfLevel, PropCover) {
  
  # --- 1. 数据准备和参数初始化 ---
  Y <- InputData$Y
  Lot <- factor(InputData$Lot)
  
  # 计算正态分布的 Z-scores
  Zbeta <- qnorm(PropCover)
  Zgamma <- qnorm(ConfLevel)
  Zbeta2 <- qnorm((1 + PropCover) / 2) # 用于双侧区间
  
  
  # --- 2. 计算中间统计量 (Recreate Table I) ---
  I <- length(unique(Lot))      # 批次数 (number of lots)
  s <- I - 1                     # 批次自由度 (lot df)
  
  J_i <- tapply(Y, Lot, length)  # 每个批次的重复测量次数
  r <- sum(J_i) - I              # 误差自由度 (error df)
  
  J_H <- I / sum(1 / J_i)        # 重复次数的调和平均数
  
  Ybar_i <- tapply(Y, Lot, mean) # 各批次的均值
  Ybar_star <- mean(Ybar_i)      # 所有批次均值的均值 (unweighted)
  
  S_sq_L <- var(Ybar_i)          # 批次均值的样本方差
  
  Yvar_i <- tapply(Y, Lot, var)  # 各批次的方差
  # 均方误差 (Mean Squared Error), na.rm=TRUE 用于处理只有1个观测值的批次
  S_sq_E <- sum((J_i - 1) * Yvar_i, na.rm = TRUE) / r
  
  Var_hat_Ybar_star <- S_sq_L / I     # Ybar_star 的估计方差
  Var_hat_Y <- S_sq_L + (1 - 1 / J_H) * S_sq_E # Y 的估计方差
  n_E <- Var_hat_Y / Var_hat_Ybar_star         # 有效样本量
  
  
  # --- 3. MLS 方法计算 Y 方差的上限 ---
  # (Modified Large-Sample method)
  alpha <- 1 - ConfLevel
  H1 <- s / qchisq(alpha, s) - 1
  H2 <- r / qchisq(alpha, r) - 1
  
  # Y 方差的上限
  U <- Var_hat_Y + sqrt((H1 * S_sq_L)^2 + (H2 * (1 - 1 / J_H) * S_sq_E)^2)
  
  
  # --- 4. 计算各种耐受区间 ---
  
  # 4.1 单侧耐受区间 (One-sided interval based on Hoffman 2010)
  L1 <- Ybar_star - Zbeta * sqrt(U) - Zgamma * sqrt(Var_hat_Ybar_star)
  U1 <- Ybar_star + Zbeta * sqrt(U) + Zgamma * sqrt(Var_hat_Ybar_star)
  
  # 4.2 双侧耐受区间 (Two-sided interval based on Hoffman & Kringle 2005)
  L2 <- Ybar_star - Zbeta2 * sqrt(1 + 1 / n_E) * sqrt(U)
  U2 <- Ybar_star + Zbeta2 * sqrt(1 + 1 / n_E) * sqrt(U)
  
  # 4.3 单侧耐受区间 (使用 HK1 调整)
  L1.HK1 <- Ybar_star - Zbeta * sqrt(1 + 1 / n_E) * sqrt(U)
  U1.HK1 <- Ybar_star + Zbeta * sqrt(1 + 1 / n_E) * sqrt(U)
  
  
  # --- 5. 格式化并返回结果 ---
  NoSlope.summary <- cbind(L1, U1, L2, U2, L1.HK1, U1.HK1)
  
  # 为结果矩阵添加列名，方便理解
  colnames(NoSlope.summary) <- c(
    "OneSided_Lower", "OneSided_Upper",
    "TwoSided_Lower", "TwoSided_Upper",
    "OneSided_HK1_Lower", "OneSided_HK1_Upper"
  )
  
  return(NoSlope.summary)
}
