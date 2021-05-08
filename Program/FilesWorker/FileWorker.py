# 根据需求获取文件路径
def get_path(tab, filename):
    if tab == 'r':
        return "../DataFiles/" + filename
    if tab == 'w':
        return "../ResultFiles/" + filename


# 读取文件中的记录并进行初步处理（直接使用spark内置文本读取器直接读取为rdd）
def read_data(spark, filename):
    rdd = spark.sparkContext.textFile(get_path('r', filename))
    # 获取文件头并且过滤掉文件头
    header = rdd.first()
    rdd = rdd.filter(lambda row: row != header)

    def map_fuc(row):
        row = row.split(',')
        # 将字符串中的带小数点的预先转换为浮点型
        for i in range(len(row)):
            if '.' in row[i]:
                row[i] = float(row[i])
        # 将记录中的字符型数据转换为可被读取的数字
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

        ret = []

        for i in range(1, len(row)):
            ret.append(int(row[i]))

        return ret

    rdd = rdd.map(map_fuc)
    # 此处为对表头进行初步过滤，原因是会造成rdd里的内容异常

    return header, rdd


# 获取文件记录并进行初步处理（使用python内置读取为列表，用来创建可并发的rdd）
def read_list_data(spark, filename):
    line_list = []
    # 读取文件记录并过滤表头
    with open(mode='r',file=get_path('r',filename)) as f:
        header = f.readline().strip()
        for line in f:
            line_list.append(line.strip().split(','))

    rdd = spark.sparkContext.parallelize(line_list)

    def map_fuc(row):
        # 将字符串中的带小数点的预先转换为浮点型
        for i in range(len(row)):
            if '.' in row[i]:
                row[i] = float(row[i])
        # 将记录中的字符型数据转换为可被读取的数字
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

        ret = []

        for i in range(1, len(row)):
            ret.append(int(row[i]))

        return ret

    rdd = rdd.map(map_fuc)
    # 初步过滤表头
    header = list(header.split(','))
    header.pop(0)

    return header, rdd.cache()


# 输出预测结果
def write_file(data,filename):
    path = get_path('w',filename)
    data = data.collect()
    with open(path,'w') as f:
        f.write('id,Response\r')
        for i in range(len(data)):
            line = str(i+1)+','+str(data[i]["prediction"])+'\r'
            f.write(line)

