import time
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier


# 根据提供的参数使用超参数验证选择最优模型
def model_training(training_data, param_info):
    # 获取参数表以及gbt模型
    param_grid, rf = model_setting(param_info)
    # 建立评估器，计算模式为准确值
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction')
    # 建立超参数验证模型
    tvs = TrainValidationSplit(estimator=rf, estimatorParamMaps=param_grid,evaluator=evaluator, trainRatio=0.8)
    # 训练模型
    model = tvs.fit(dataset=training_data)
    # 返回最优模型
    return model.bestModel


# 根据提供的参数建立模型及其参数表
def model_setting(param_info):
    # 建立gbt模型
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol='weight')
    # 获取参数列表
    bin_info = get_param(param_info)
    # 建立参数表
    param_grid = ParamGridBuilder().addGrid(rf.maxBins, bin_info).build()
    return param_grid, rf


# 根据第一次训练模型返回的参数建立参数列表
def get_param(param_info):
    bin_info = []
    param_info -= 5
    for i in range(10):
        bin_info.append(param_info)
        param_info += 1
    return bin_info


# 根据提供的参数建立模型并训练以及评估，最后返回其参数
def model_learning(training_data, test_data, num_bins=25):
    # 建立gbt模型
    time_start = time.time()    # 计时开始
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol='weight',
                                maxBins=num_bins)
    model = rf.fit(training_data)
    time_end = time.time()  # 计时结束
    print('训练时间=' + str(time_end - time_start) + 's')  # 打印训练模型花费的时间
    # 评估模型
    time_start = time.time()    # 计时开始
    Evaluator(model,test_data)
    time_end = time.time()  # 计时结束
    print('验证时间=' + str(time_end - time_start) + 's')  # 打印训练模型花费的时间
    # 获取参数并返回
    return model.getOrDefault('maxBins')


# 建立评估器并评估模型（评估其精确度、召回率、ROC指标以及f1分数）
def Evaluator(model, test_data):
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction')
    mu_evalutor = MulticlassClassificationEvaluator(metricName='f1')
    TP = float(predictions.select('label','prediction').rdd.filter(lambda x:x[0] == 1.0 and x[1] == 1.0).count())
    FN = float(predictions.select('label','prediction').rdd.filter(lambda x:x[0] == 1.0 and x[1] == 0.0).count())
    FP = float(predictions.select('label','prediction').rdd.filter(lambda x:x[0] == 0.0 and x[1] == 1.0).count())
    print("Precision = %f " % (TP/(TP+FP)))
    print("Recall = %f " % (TP/(TP+FN)))
    print("ROC = %f" % evaluator.evaluate(predictions))
    print("f1 = %f" % mu_evalutor.evaluate(predictions))
