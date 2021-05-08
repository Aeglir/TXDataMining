import Program.PCAPLOT.PCAPLOT as pp
from pyspark.ml.feature import UnivariateFeatureSelector, MinMaxScaler, VectorIndexer
from pyspark.ml.linalg import Vectors
import numpy as np
import Program.FilesWorker.FileWorker as fw
from pyspark.sql.functions import when, col


# 准备用于训练的数据
def training_data_prepare(spark, filename):
    # 若只是处理较少数据时，为提升速度应启用该函数
    header, rdd = fw.read_data(spark, filename)
    # 初步过滤表头
    header = list(header.split(','))
    header.pop(0)

    # # 若要进行并发应该启用该函数
    # header, rdd = fw.read_list_data(spark, filename)
    # 打印数据的记录数、最大值、最小值、平均值以及标准差，以及计算皮尔逊相关系数
    personal_array = data_description(spark, rdd.cache(), header)
    # 打印数据的皮尔逊相关系数
    personal_show(personal_array, header)
    # 将文件中数量级过大的数据转换为合适的规格
    rdd = annual_premium_scaler(spark, rdd, header)
    # 将线性相关性强的特征进行合并
    rdd, header = damaged_couple(rdd, header)
    # 将年龄进行定性化
    rdd = arrange_for_age(rdd, header)
    # 使用卡方验证二次清洗数据
    cleaned_rdd, cleaned_header = useful_select(spark, rdd, header)

    def map_fuc_rdd(row):
        ret = []
        for i in range(len(row) - 1):
            ret.append(row[i])

        return ret, row[-1]

    cleaned_rdd = cleaned_rdd.map(map_fuc_rdd)

    # 设置PCA降维的维度
    # n = 3
    # pp.PCA_builder(spark,rdd,n)
    # 将数据格式转换为机器学习所要求的格式
    def map_fuc(row):
        features_array = np.array(row[0])
        index_array = np.arange(features_array.size)
        num = features_array.size

        return row[1], Vectors.sparse(num, index_array, features_array)

    labeled_points_rdd = cleaned_rdd.map(map_fuc)
    # print(labeled_points_rdd.first())
    data = spark.createDataFrame(labeled_points_rdd, schema=['label', 'indexedFeatures'])
    # 使用特征转换器进行对数据的进一步处理
    data = VectorIndexer(inputCol="indexedFeatures", outputCol="features",
                         maxCategories=4).fit(data).transform(data)
    # 筛选掉无用的列
    data = data.drop("indexedFeatures")
    data = balanceDataset(data)
    # 将数据集分为训练集和测试集
    training_data, check_data = data.randomSplit([0.7, 0.3])
    # 对小类进行过采样
    training_data = enlarge_data(training_data, 0.15)
    return training_data.cache(), check_data.cache(), cleaned_header


# 对小类进行过采样
def enlarge_data(df, percent=0.5):
    sample_data = df.filter(df.label == 1).sample(True, percent)
    return df.union(sample_data)


# 将Previously_Insured和Vehicle_Damage呈现负线性相关的两项进行组合
def damaged_couple(rdd, head):
    num_previously_insured = 0
    num_vehicle_damage = 0
    for i in range(len(head)):
        if head[i] == 'Previously_Insured':
            num_previously_insured = i
        if head[i] == 'Vehicle_Damage':
            num_vehicle_damage = i

    def map_fuc(row):
        ret = []
        for ch in row:
            ret.append(ch)

        ret[num_previously_insured] = (2 * ret[num_vehicle_damage] - ret[num_previously_insured]) * 2
        ret.pop(num_vehicle_damage)
        return ret

    rdd = rdd.map(map_fuc)
    head.pop(num_vehicle_damage)

    return rdd, head


# 将年龄进行定性化
def arrange_for_age(rdd, header):
    for i in range(len(header)):
        if header[i] == 'Age':
            num_age = i

    def map_fuc(row):
        ret = []
        for ch in row:
            ret.append(ch)

        ret[num_age] = int(ret[num_age] / 10) * 10
        return ret

    rdd = rdd.map(map_fuc)
    return rdd


# 将文件中数量级过大的数据转换为合适的规格
def annual_premium_scaler(spark, rdd, header):
    num_annual_premium = 0
    for i in range(len(header)):
        if header[i] == 'Annual_Premium':
            num_annual_premium = i

    # 将数据中需要进行标准化的记录提取出来
    def map_annual_vector(row):
        ret = []
        for i in range(len(row)):
            if i is num_annual_premium:
                ret.append(Vectors.dense(row[i]))
            else:
                ret.append(row[i])
        return ret

    # 提取需要标准化的记录
    annual_vectors = rdd.map(map_annual_vector)
    df = spark.createDataFrame(annual_vectors, schema=header)
    # 使用MinMaxScaler标准化数据
    mm_scaler = MinMaxScaler(inputCol='Annual_Premium', outputCol="Annual_Premium_Scaler", min=0, max=200)
    model = mm_scaler.fit(df)

    # 将标准化后的列替代掉原来过大的数据所在的列
    def map_fuc(row):
        ret = []
        for i in range(len(row) - 1):
            if i is num_annual_premium:
                ret.append(int(row[len(row) - 1].toArray()[0]))
            else:
                ret.append(row[i])

        return ret

    # 得到标准化完毕的rdd
    transform_list = model.transform(df).rdd.map(map_fuc)
    return transform_list


# 打印皮尔逊相关系数矩阵
def personal_show(personal_array, header):
    print(header)
    for row in personal_array:
        print(row)


# 打印数据的记录数、最大值、最小值、平均值以及标准差，以及计算皮尔逊相关系数
def data_description(spark, rdd, header):
    df = spark.createDataFrame(rdd, schema=header)
    # 生成含有数据的记录数、最大值、最小值、平均值以及标准差的表格
    summary_df = df.summary("count", "max", "min", "mean", "stddev")
    summary_df.show()
    # 过滤计算皮尔逊相关系数时使用不到的记录
    sdf = summary_df.collect()
    sdf.pop(0)
    sdf.pop(0)
    sdf.pop(0)
    # 计算皮尔逊相关系数
    personal_array = personal_calculator(sdf, rdd)
    return personal_array


# 计算皮尔逊相关系数
def personal_calculator(summary, rdd):
    num_count = rdd.count()
    # 需要计算皮尔逊相关系数的特征数
    num_row = len(summary[0]) - 2

    # 将每一记录减去其标准差
    def map_fuc(row):
        ret = []
        for i in range(num_row):
            ret.append(row[i] - float(summary[0][i + 1]))
        return ret

    # 转换为列表便于读取
    mean_list = rdd.map(map_fuc).collect()
    # 计算皮尔逊系数的分子部分
    personal_array = []
    for i in range(num_row):
        personal_num = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for j in range(num_row):
            for ROW in mean_list:
                personal_num[j] += ROW[j] * ROW[i]
        personal_array.append(personal_num)

    # 计算皮尔逊系数的分子部分的期望值
    def array_map_fuc(row):
        return list(map(lambda x: x / num_count, row))

    personal_array = list(map(array_map_fuc, personal_array))
    # 将分子部分除以其分母，完成计算
    for i in range(num_row):
        for j in range(num_row):
            personal_array[i][j] /= float(summary[1][j + 1]) * float(summary[1][i + 1])

    return personal_array


# 使用卡方检验进一步清洗数据
def useful_select(spark, rdd, header):
    # 将rdd转换为指定的格式
    def vectors_map_fuc(row):
        vector = []
        for i in range(len(row) - 1):
            vector.append(row[i])

        vector = Vectors.dense(vector)
        return vector, row[-1]

    vectors_rdd = rdd.map(vectors_map_fuc)
    # 添加表头
    df = spark.createDataFrame(vectors_rdd, schema=["features", "label"])
    # 设置卡方选择器的相关设置
    selector = UnivariateFeatureSelector(outputCol="selectedFeatures", selectionMode='fpr')
    # 设置卡方验证的最小值
    max_p = 0.5
    # 使用卡方选择器筛选数据
    selector.setFeatureType("categorical").setLabelType("categorical").setSelectionThreshold(max_p)
    model = selector.fit(df)
    result = model.selectedFeatures

    # 使用卡方选择的结果过滤数据
    def map_fuc(row):
        ret = []
        for ch in result:
            ret.append(row[ch])

        ret.append(row[-1])
        return ret

    rdd = rdd.map(map_fuc)
    # 根据结果进一步清洗表头
    cleaned_header = []
    for i in range(len(header)):
        if i in result:
            cleaned_header.append(header[i])
    cleaned_header.append(header[-1])
    return rdd, cleaned_header


# 准备需要预测的数据
def test_data_prepare(spark, filename, header):
    # 若只是处理较少数据时，为提升速度应启用该函数
    test_header, rdd = fw.read_data(spark, filename)
    # 初步清洗表头
    test_header = list(test_header.split(','))
    test_header.pop(0)

    # # 若要进行并发应该启用该函数
    # test_header, rdd = fw.read_list_data(spark, filename)
    # 将文件中数量级过大的数据转换为合适的规格
    rdd = annual_premium_scaler(spark, rdd, test_header)
    # 将相关性强的特征进行合并
    rdd, test_header = damaged_couple(rdd, test_header)
    # 将年龄进行定性化
    rdd = arrange_for_age(rdd, test_header)
    # 根据清洗后的训练集的表头筛选要保留的列并将其记录到列表中
    num_array = []
    for i in range(len(test_header)):
        if test_header[i] in header:
            num_array.append(i)

    # 根据列表过滤数据
    def map_fuc(row):
        ret = []
        for n in num_array:
            ret.append(row[n])

        return ret

    rdd = rdd.map(map_fuc)

    # 将格式转换为机器学习算法所需要的格式
    def Vectors_map_fuc(row):
        features_array = np.array(row)
        index_array = np.arange(features_array.size)
        num = features_array.size

        return (Vectors.sparse(num, index_array, features_array),)

    labeled_points_rdd = rdd.map(Vectors_map_fuc)
    data = spark.createDataFrame(labeled_points_rdd, schema=['indexedFeatures'])
    data = VectorIndexer(inputCol="indexedFeatures", outputCol="features",
                         maxCategories=4).fit(data).transform(data)
    # 过滤掉不需要的表头
    data = data.drop("indexedFeatures")
    return data.cache()


# 设置权重
def balanceDataset(df):
    # 计算权重
    num_positives = df.filter(df['label'] == 1).count()
    num_data = df.count()
    balancing_ratio = float(num_positives) / num_data
    # 添加权重列
    weighted_df = df.withColumn('weight', when(col('label').isin(0), balancing_ratio).otherwise(1 - balancing_ratio))
    return weighted_df
