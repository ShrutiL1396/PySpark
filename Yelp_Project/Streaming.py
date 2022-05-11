from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col

spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .getOrCreate()

# Create DataFrame representing the stream of input lines from connection to localhost:9999
lines = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load().withColumnRenamed("value","text")


sameModel = PipelineModel.load("./elite_model.model")
predictions = sameModel.transform(lines).select('text','prediction')

 # Start running the query that prints the predictions
query = predictions \
    .writeStream \
    .format("console") \
    .start()

query.awaitTermination()
