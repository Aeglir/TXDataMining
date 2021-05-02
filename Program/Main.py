import time
from pyspark.sql import SparkSession
from Program.DataPreparer.DataPreparer import training_data_prepare, test_data_prepare
from Program.ModelBuilder.ModelBuilder import model_training, model_learning, Evaluator
from Program.FilesWorker.FileWorker import write_file

if __name__ == "__main__":
    # 设置好SPARK的设置（内存，核数，并行数量）在单机情况下增加并发量会显著减慢程序运行速度并显著提升处理数据的最大上限
    spark = SparkSession.builder \
        .master("local") \
        .appName("Word Count") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.default.parallelism", 200) \
        .getOrCreate()

    train_filename = "small_debug.csv"       # 训练模型用的文件名称
    test_filename = "debug_test.csv"   # 包含需要模型预测的数据的文件名称
    out_filename = "predictions.csv"   # 预测结果输出文件名称

    # 获取训练数据、检验数据和文件头
    training_data, check_data, header = training_data_prepare(spark, train_filename)
    test_data = test_data_prepare(spark, test_filename, header)  # 将预测文件处理成指定的格式

    time_start = time.time()    # 计时开始
    param_info = model_learning(training_data, check_data)  # 训练并检验模型
    time_end = time.time()  # 计时结束
    print('时间=' + str(time_end - time_start) + 's')  # 打印训练模型花费的时间
    # print(param_info)

    best_model = model_training(training_data, param_info)  # 根据超参数获取模型的最优参数
    Evaluator(best_model, check_data)   # 评估最优参数的模型的命中率

    write_file(best_model.transform(test_data).select("prediction").rdd, out_filename)  # 获取预测并将预测输出
