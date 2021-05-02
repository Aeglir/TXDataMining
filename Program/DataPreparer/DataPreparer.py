# import Program.PCAPLOT.PCAPLOT as pp
from pyspark.ml.feature import UnivariateFeatureSelector, MinMaxScaler
from pyspark.ml.linalg import Vectors


def data_prepare(spark, rdd, header):
    header = list(header.split(','))

    first_header = header.pop(0)

    rdd = annual_premium_scaler(spark, rdd, header)

    personal_array = data_description(spark, rdd, header)

    # print(personal_array)

    # personal_show(personal_array, header)

    # n = 3
    # pp.PCA_builder(spark,rdd,n)

    cleaned_rdd, cleaned_header = clean_data(rdd, personal_array, header)

    # print(cleaned_header)
    # print(cleaned_rdd.first())

    # print(cleaned_header)
    cleaned_rdd, cleaned_header = useful_select(spark,cleaned_rdd,cleaned_header)

    return cleaned_rdd,cleaned_header


def annual_premium_scaler(spark, rdd, header):
    def map_annual_vector(row):
        return row[0], row[1], row[2], row[3], row[4], row[5], row[6], Vectors.dense(row[7]), row[8], row[9], row[10]

    annual_vectors = rdd.map(map_annual_vector)

    df = spark.createDataFrame(annual_vectors, schema=header)

    mmScaler = MinMaxScaler(inputCol='Annual_Premium', outputCol="Annual_Premium_Scaler", min=0, max=100)

    model = mmScaler.fit(df)

    # print(annual_vectors.take(10))

    # print(model.originalMin)
    # print(model.originalMax)

    def map_fuc(row):
        return row[0], row[1], row[2], row[3], row[4], row[5], row[6], int(row[11].toArray()[0]), row[8], row[9], row[10]


    transform_list = model.transform(df).rdd.map(map_fuc)

    # print(transform_list.first())

    return transform_list


def personal_show(personal_array, header):
    print(header)
    for row in personal_array:
        print(row)


def data_description(spark, rdd, header):
    df = spark.createDataFrame(rdd, schema=header)

    summary_df = df.summary("count", "max", "min", "mean", "stddev")

    summary_df.show()

    sdf = summary_df.collect()

    sdf.pop(0)
    sdf.pop(0)
    sdf.pop(0)

    personal_array = personal_calculator(sdf, rdd)

    return personal_array


def personal_calculator(summary, rdd):
    num_count = rdd.count()
    num_row = len(summary[0]) - 2

    def map_fuc(row):
        ret = []
        for i in range(num_row):
            ret.append(row[i] - float(summary[0][i + 1]))

        return ret

    mean_list = rdd.map(map_fuc).collect()

    # print(mean_list)

    personal_array = []

    for i in range(num_row):
        personal_num = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for j in range(num_row):
            for ROW in mean_list:
                personal_num[j] += ROW[j] * ROW[i]
        personal_array.append(personal_num)

    def array_map_fuc(row):
        return list(map(lambda x: x / num_count, row))

    personal_array = list(map(array_map_fuc, personal_array))

    for i in range(num_row):
        for j in range(num_row):
            personal_array[i][j] /= float(summary[1][j + 1]) * float(summary[1][i + 1])

    return personal_array


def clean_data(rdd, personal_array, header):
    max_personal_num = 0.75

    select = []

    for i in range(len(personal_array)):
        t = True
        for j in range(len(personal_array)):
            if j > i:
                if abs(personal_array[i][j]) > max_personal_num:
                    t = False
            else:
                continue
        if t:
            select.append(i)

    def map_fuc(row):
        ret = []

        for num in select:
            ret.append(row[num])

        ret.append(row[len(personal_array)])

        return ret

    rdd = rdd.map(map_fuc)

    head = []

    for x in select:
        head.append(header[x])

    head.append(header[len(header)-1])

    return rdd, head


def useful_select(spark, rdd, header):
    def vectors_map_fuc(row):
        vector = []
        for i in range(len(row) - 1):
            vector.append(row[i])

        vector = Vectors.dense(vector)

        return vector, row[len(row) - 1]

    vectors_rdd = rdd.map(vectors_map_fuc)

    df = spark.createDataFrame(vectors_rdd, schema=["features", "label"])

    selector = UnivariateFeatureSelector(outputCol="selectedFeatures", selectionMode='fpr')

    max_p = 0.5

    selector.setFeatureType("categorical").setLabelType("categorical").setSelectionThreshold(0.5)

    model = selector.fit(df)

    result = model.selectedFeatures

    def map_fuc(row):
        ret = []
        for ch in result:
            ret.append(row[ch])

        return tuple(ret),row[len(row)-1]

    cleaned_header = []

    for i in range(len(header)):
        if i in result:
            cleaned_header.append(header[i])

    cleaned_header.append(header[len(header)-1])

    rdd = rdd.map(map_fuc)

    # print(result)
    # print(cleaned_header)
    # print(rdd.first())

    return rdd,cleaned_header
