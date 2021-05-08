import matplotlib.pyplot as pl
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import PCA


def PCA_builder(spark,rdd,n):
    vectors_rdd = tran_vectors(rdd)

    if n == 2:
        print_2D(PCA_setting(spark,vectors_rdd,n))
    elif n == 3:
        print_3D(PCA_setting(spark,vectors_rdd,n))


def tran_vectors(rdd):

    def map_fuc(row):
        ret = []
        for ch in row[0]:
            ret.append(ch)
        return (Vectors.dense(ret),)

    return rdd.map(map_fuc)


def PCA_setting(spark, rdd, n):
    df = spark.createDataFrame(rdd,schema=['features'])
    pca = PCA(k=n,inputCol='features',outputCol='pca_features')

    model = pca.fit(df)

    return model.transform(df).select('pca_features').collect()


def print_2D(df):
    pl.title("visualization of Data in 2-dim")
    x = []
    y = []

    for dv in df:
        x.append(dv[0][0])
        y.append(dv[0][1])

    pl.scatter(x,y)

    pl.show()


def print_3D(df):
    pl.title("visualization of Data in 2-dim")
    x = []
    y = []
    z = []

    ax = pl.subplot(111, projection='3d')

    for dv in df:
        x.append(dv[0][0])
        y.append(dv[0][1])
        z.append(dv[0][2])

    ax.scatter(x,y,z)

    pl.show()
