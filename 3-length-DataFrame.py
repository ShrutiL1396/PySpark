from pyspark import SparkContext
from pyspark import sql
from pyspark.sql import SQLContext
from pyspark.sql.functions import avg
import re

sc = SparkContext()
sqlContext = sql.SQLContext(sc)

#Loading Data from file
dataRDD = sc.textFile("Amazon_Comments.csv").map(lambda x:x.split("^"))

#Find length of review
dataRDDnew = dataRDD.map(lambda x:(x[0],x[1],x[2],x[3],x[4],x[5],x[6],len(re.sub('W+',' ',x[5]).strip().split(' '))))

#Loadning Data into SparkDataFrame
dataDF = dataRDDnew.toDF(["ProductID","ReviewID", "ReviewTitle","ReviewTime","Verified","Review","Rating","Length"])

#####################################
#Using Dataframe  SQL to find average length by Rating

results = dataDF.groupBy("Rating").avg("Length")

#####################################

#Create list of results
answer = sorted(results.collect())

#Print Output
for i in answer:
	print (str(i[0])+ " Star Rating: Average Length of Comments "+ str(i[1]))
