from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.sql.functions import lit
from pyspark.sql import Window


sc = SparkContext("local[*]")
spark = SparkSession(sc)

#"review_id","user_id","business_id","stars","date","text","useful","funny","cool"

#1. Clean the dataset

df = spark.read.csv( "yelp_review.csv", header=True)
#df = df.withColumnRenamed("stars", "label")
df = df.withColumn("label", df["stars"].cast("double"))
#df = df.where(col("label").isNotNull())
df = df.dropna(subset=['label', 'text', 'funny', 'cool',"useful"])

df = df.select('text', 'label')

df_unlab = df.select('text')
df_unlab = df_unlab.limit(10)

# df = df.filter(df.label.isin(2.0,4.0))
# df.show()

df = df.limit(1000)

#2 ML pipeline
tokenizer = Tokenizer(inputCol = "text", outputCol = "words") 
TF = HashingTF(inputCol = tokenizer.getOutputCol(), outputCol = "tfFeatures")
idf = IDF(inputCol = "tfFeatures",outputCol = "features")
idf.show
lr = LogisticRegression(maxIter = 10, regParam = 0.001)

pipeline = Pipeline(stages = [tokenizer,TF,idf,lr])
model = pipeline.fit(df)

#####################################
#3. Test data set, unlabeled text, your code here:
test = df_unlab

# #4. Make Prediction
prediction = model.transform(test)

selected = prediction.select("text","probability","prediction")
for row in selected.collect():
	text, prob, prediction = row
	print("( %s --> prediction = %f" %(text, prediction))

#####################################
