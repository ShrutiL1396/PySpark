from pyspark import SparkConf, SparkContext
import nltk
import re 

# creating the PySpark instance, using all the available processors locally
conf = SparkConf().setMaster("local[*]").setAppName("1-length")
sc = SparkContext(conf = conf)

# importing nltk to set stopword which includes all the commonly used english stopwords, such as articles, 'the'
# 'a', 'it',etc.
from nltk.corpus import stopwords
stopword_list = set(stopwords.words("english"))

# load file and create rdd from it
lines = sc.textFile('Amazon_Comments.txt')

# extrating the comment and rating from the rdd which consists of all reviews.
line_punct = lines.map(lambda x: (x.split("^")[-2],x.split("^")[-1]))

# creating an rdd each for each rating
line_5s = line_punct.filter(lambda x: x[1] == '5.00')
line_4s = line_punct.filter(lambda x: x[1] == '4.00')
line_3s = line_punct.filter(lambda x: x[1] == '3.00')
line_2s = line_punct.filter(lambda x: x[1] == '2.00')
line_1s = line_punct.filter(lambda x: x[1] == '1.00')

# creating a function ProcessText, which will remove the stopwords from each of the reviews through their
# respective RDD
def ProcessText(text):
	tokens = nltk.word_tokenize(text)
	remove_punct = [word for word in tokens if word.isalpha()]
	remove_stop_words = [word for word in remove_punct if not word in stopword_list]
	return remove_stop_words

words_5 = line_5s.flatMap(lambda x: ProcessText(str(x).lower()))
words_4 = line_4s.flatMap(lambda x: ProcessText(str(x).lower()))
words_3 = line_3s.flatMap(lambda x: ProcessText(str(x).lower()))
words_2 = line_2s.flatMap(lambda x: ProcessText(str(x).lower()))
words_1 = line_1s.flatMap(lambda x: ProcessText(str(x).lower()))

# creating an RDD for each rating which comprises of tuples. Each tuple has 2 elements, first being the 
# word and second being the value '1' as this will further help us in getting the individual word frequency for each rating.
words_5fm = words_5.map(lambda x:(x,1))
words_4fm = words_4.map(lambda x:(x,1))
words_3fm = words_3.map(lambda x:(x,1))
words_2fm = words_2.map(lambda x:(x,1))
words_1fm = words_1.map(lambda x:(x,1))

# Word frequency using reduceByKey()

result_5 = words_5fm.reduceByKey(lambda x,y:x+y)
result_4 = words_4fm.reduceByKey(lambda x,y:x+y)
result_3 = words_3fm.reduceByKey(lambda x,y:x+y)
result_2 = words_2fm.reduceByKey(lambda x,y:x+y)
result_1 = words_1fm.reduceByKey(lambda x,y:x+y)

# sorting the above RDDs in descending order of frequency count and extracting only the top 10 common words in each rating
sorted_5 = (sorted(result_5.collect(),key = lambda x:x[1],reverse=True)[:10])
sorted_4 = (sorted(result_4.collect(),key = lambda x:x[1],reverse=True)[:10])
sorted_3 = (sorted(result_3.collect(),key = lambda x:x[1],reverse=True)[:10])
sorted_2 = (sorted(result_2.collect(),key = lambda x:x[1],reverse=True)[:10])
sorted_1 = (sorted(result_1.collect(),key = lambda x:x[1],reverse=True)[:10])

# extrating only the word from the RDD
sorted_5_b = sc.parallelize(sorted_5)
sorted_5_words = sorted_5_b.map(lambda x: x[0])
sorted_4_b = sc.parallelize(sorted_4)
sorted_4_words = sorted_4_b.map(lambda x: x[0])
sorted_3_b = sc.parallelize(sorted_3)
sorted_3_words = sorted_3_b.map(lambda x: x[0])
sorted_2_b = sc.parallelize(sorted_2)
sorted_2_words = sorted_2_b.map(lambda x: x[0])
sorted_1_b = sc.parallelize(sorted_1)
sorted_1_words = sorted_1_b.map(lambda x: x[0])


print("1 star rating: ",sorted_1_words.collect())
print("2 star rating: ",sorted_2_words.collect())
print("3 star rating: ",sorted_3_words.collect())
print("4 star rating: ",sorted_4_words.collect())
print("5 star rating: ",sorted_5_words.collect())

