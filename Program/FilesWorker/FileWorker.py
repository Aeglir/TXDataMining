


def get_path(tab, filename):
    if tab == 'r':
        return "../DataFiles/" + filename + ".csv"
    if tab == 'w':
        return "../ResultFiles/" + filename + ".csv"


def read_data(spark, filename):
    rdd = spark.sparkContext.textFile(get_path('r', filename))
    header = rdd.first()
    rdd = rdd.filter(lambda row: row != header)

    def map_fuc(row):
        row = row.split(',')
        if row[1] == 'Male':
            row[1] = 1
        elif row[1] == 'Female':
            row[1] = 0

        if row[6] == '< 1 Year':
            row[6] = 0
        elif row[6] == '1-2 Year':
            row[6] = 1
        elif row[6] == '> 2 Years':
            row[6] = 2

        if row[7] == 'Yes':
            row[7] = 1
        elif row[7] == 'No':
            row[7] = 0

        row[2] = int(row[2])
        row[3] = int(row[3])
        row[4] = int(row[4])
        row[5] = int(row[5])
        row[8] = int(row[8])
        row[9] = int(row[9])
        row[10] = int(row[10])
        row[11] = int(row[11])

        return row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]

    rdd = rdd.map(map_fuc)

    return header, rdd

