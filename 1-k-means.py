from pyspark.context import SparkContext, SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors
# Started Spark Context with Spark Session 
conf = SparkConf().setMaster("local").setAppName("1-k-means")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

"""
Using MLlib Library
"""
# Read in the txt file
read_data = sc.textFile('kmeans_data.txt')
# Split each line into a list seperated by " "
read_data = read_data.map(lambda x: x.split(" "))
# Converts each line into a Vectors containing just the X and Y columns
data_lib = read_data.map(lambda x: [Vectors.dense(float(x[1]), float(x[2]))])
# turning our data into a Data Frame with columns: features
df = spark.createDataFrame(data_lib, ["features"])

#####################################

# initialize the model with K = 2 
kmeans = KMeans(k=2)

# Fit the model with our Data Frame
model = kmeans.fit(df)

# Grab all the centers
ctr = []
centers = model.clusterCenters()

# will print out all of the centers 

print("Cluster Centers: ")
for center in centers:
    ctr.append(center)
    print(center)
