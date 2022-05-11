<h1 align="center">Data Analysis using PySpark and Hadoop</h1>

Big Data is often extremely voluminous and rich in information. However traditional Data Analysis tools often fail to handle this plethora of data which can contain Data Streams as well. In order to perform wrangling and insight gathering on such datasets we need to make use of PySpark.

## Description
This repository attempts to showcase Data Streaming using Python provided "Generators", "Iterators" and High Order Functions such as filter(), map() and reduce(). Generators can be used as Data Streams and usually do not store everything in memory, a vital requirement when dealing with Big Data. The exercises in this repository show the usage of PySpark RDDs or Resilient Distributed Dataframes which perform operations on datasets by leveraging Hadoop backed Distributed Parallel computing. Spark Data frames are also a more user-friendly mode of performing data manipulation of structured data. Machine Learning and exercises which run Spark on Google Cloud Platform's Dataproc cluster have also been covered in this repository.

## Exercises
1. Compute the median age of the Citibike's subscribed customers using generators (1-generator.ipynb).
2. Implement Python's High Order Functions (2-HOF.ipynb).
3. Extract the first ride of the day from a Citibike data stream (3-generator-HOF.ipynb) (Citibike.csv used in all above exercises).
4. Calculate average length of Amazon reviews for each type of rating (1 through 5) using RDDs (1-length.py).
5. Pick top 10 occuring words in comments of each rating type using RDDs (2-wordranking.py) (Amazon_Comments.txt file used in 4. and 5.).
6. Implement k-means clustering algorithm in PySpark (1-k-means.py).
7. Implement logistic regression to predict the rating of first 10 comments (2-lr.py).
8. Calculate average length of Amazon reviews for each type of rating (1 through 5) using Spark Dataframes (3-length-DataFrame.py).

## Prerequisites and Installation
 - pyspark
 ```
 pip install pyspark
 ```
 
 - pyspark ML
 - SparkConf
 - SparkContext
 - Google Cloud Account
 
 ## Contents
[PySpark Exercises](https://github.com/ShrutiL1396/PySpark) <br/>


## Contact
Shruti Shivaji Lanke - <br/>
shrutilanke13@gmail.com or slanke1@student.gsu.edu <br/>
Project Link - <br/>
https://github.com/ShrutiL1396/PySpark
