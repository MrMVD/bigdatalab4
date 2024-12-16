from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Инициализация спарка
spark = SparkSession.builder \
    .appName("Salary Prediction") \
    .getOrCreate()

# Загрузка данных
file_path = "../data/Employee_Salaries_-_2023.csv"  
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Создание двоичной целевой переменной
data = data.withColumn("Above_50K", when(col("Base_Salary") > 50000, 1).otherwise(0))

# Обработка категориальных переменных
division_indexer = StringIndexer(inputCol="Division", outputCol="DivisionIndex")
department_indexer = StringIndexer(inputCol="Department_Name", outputCol="DepartmentIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")

# В числа
division_encoder = OneHotEncoder(inputCol="DivisionIndex", outputCol="DivisionVec")
department_encoder = OneHotEncoder(inputCol="DepartmentIndex", outputCol="DepartmentVec")

# Вектор признаков
assembler = VectorAssembler(
    inputCols=["DivisionVec","DepartmentVec", "GenderIndex"], 
    outputCol="Features"
)

# Стандартизируем признаки
scaler = StandardScaler(inputCol="Features", outputCol="ScaledFeatures")

preprocessing_pipeline = Pipeline(stages=[
    division_indexer,
    department_indexer, 
    gender_indexer, 
    division_encoder,
    department_encoder, 
    assembler, 
    scaler
])

processed_data = preprocessing_pipeline.fit(data).transform(data)   

# Разделяем данные
train_data, test_data = processed_data.randomSplit([0.8, 0.2])

# Обучение
rf = RandomForestClassifier(featuresCol="ScaledFeatures", labelCol="Above_50K")

rf_model = rf.fit(train_data)

# Оценка модели
predictions = rf_model.transform(test_data)

# Оенка точности
evaluator = MulticlassClassificationEvaluator(
    labelCol="Above_50K", 
    predictionCol="prediction", 
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Остановка спарка
spark.stop()