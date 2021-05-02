from pyspark.sql import SparkSession
import Program.FilesWorker.FileWorker as fw
import Program.DataPreparer.DataPreparer as dp

if __name__ == "__main__":
    spark = SparkSession.builder \
        .master("local") \
        .appName("Word Count") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "4") \
        .config("spark.default.parallelism", 30) \
        .getOrCreate()

    train_filename = "debug"

    header, rdd = fw.read_data(spark, train_filename)

    # dp.data_prepare(spark, rdd, header)

    clean_rdd, clean_head = dp.data_prepare(spark, rdd, header)

    print(clean_head)

    print(clean_rdd.first())
    # rdd = spark.sparkContext.parallelize(rdd.filter(lambda row: row != header).takeSample(False, 1000))

    # key = rdd.map(lambda row: row[6])

    # print(header)
