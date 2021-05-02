from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# 根据提供的参数使用超参数验证选择最优模型
def model_training(training_data, param_info):
    # 获取参数表以及gbt模型
    param_grid, gbt = model_setting(param_info)

    # 建立评估器，计算模式为准确值
    evaluator = MulticlassClassificationEvaluator(metricName='accuracy')

    # 建立超参数验证模型
    tvs = TrainValidationSplit(estimator=gbt, estimatorParamMaps=param_grid,
                               evaluator=evaluator, trainRatio=0.8)

    # 训练模型
    model = tvs.fit(dataset=training_data)

    # 返回最优模型
    return model.bestModel


# 根据提供的参数建立模型及其参数表
def model_setting(param_info):
    # 建立gbt模型
    gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

    # 获取参数列表
    bin_info,depth_info,step_size_info = get_param(param_info)

    # 建立参数表
    param_grid = ParamGridBuilder().addGrid(gbt.maxBins, bin_info).addGrid(gbt.stepSize, step_size_info) \
        .addGrid(gbt.maxDepth, depth_info).build()
    return param_grid, gbt


# 根据第一次训练模型返回的参数建立参数列表
def get_param(param_info):
    bin_info = []
    depth_info = []
    step_size_info = []
    bin_info.append(param_info[0])
    bin_info.append(param_info[0]*2)
    depth_info.append(param_info[1])
    depth_info.append(param_info[1]*2)
    step_size_info.append(param_info[2])
    step_size_info.append(param_info[2]*2)
    return bin_info,depth_info,step_size_info


# 根据提供的参数建立模型并训练以及评估，最后返回其参数
def model_learning(training_data, test_data, num_bins=4, num_depth=6, num_step_size=0.1):
    # 建立gbt模型
    gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10,
                        maxBins=num_bins, maxDepth=num_depth,stepSize=num_step_size)
    model = gbt.fit(training_data)
    # 评估模型
    Evaluator(model,test_data)
    # 获取参数并返回
    return [model.getOrDefault('maxBins'),model.getOrDefault('maxDepth'),model.getOrDefault('stepSize')]


# 建立评估器并评估模型（评估其准确值）
def Evaluator(model, test_data):
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
    print("accuracy = %f" % evaluator.evaluate(predictions))
