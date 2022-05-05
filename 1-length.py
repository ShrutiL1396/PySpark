from pyspark import SparkConf, SparkContext
import nltk
import re 

# creating the PySpark instance, using all the available processors locally 
conf = SparkConf().setMaster("local[*]").setAppName("1-length")
sc = SparkContext(conf = conf)


# load the input file as '.txt' and create rdd from it
lines = sc.textFile('Amazon_Comments.txt')

# extrating the comment and rating from the rdd which consists of all reviews.
line_punct = lines.map(lambda x: (x.split("^")[-2],x.split("^")[-1]))
line_punct.persist()

# creating an rdd each for each rating
line_5s = line_punct.filter(lambda x: x[1] == '5.00')
line_4s = line_punct.filter(lambda x: x[1] == '4.00')
line_3s = line_punct.filter(lambda x: x[1] == '3.00')
line_2s = line_punct.filter(lambda x: x[1] == '2.00')
line_1s = line_punct.filter(lambda x: x[1] == '1.00')


# getting the overall count of words for comments belonging to the said ratings
count_5 = line_5s.count()
count_4 = line_4s.count()
count_3 = line_3s.count()
count_2 = line_2s.count()
count_1 = line_1s.count()

# creating a function which gets rid off punctuation marks and other special characters 
def ProcessText(text):
    return re.sub('\W+', ' ', str(text))

line_5wo_puct = line_5s.map(lambda x:ProcessText(x))
line_4wo_puct = line_4s.map(lambda x:ProcessText(x))
line_3wo_puct = line_3s.map(lambda x:ProcessText(x))
line_2wo_puct = line_2s.map(lambda x:ProcessText(x))
line_1wo_puct = line_1s.map(lambda x:ProcessText(x))

# creating an rdd using flatMap of all the comments, of each rating, to get the total number of relevant words 
words_5 = line_5wo_puct.flatMap(lambda x:x.lower().split(" "))
words_4 = line_4wo_puct.flatMap(lambda x:x.lower().split(" "))
words_3 = line_3wo_puct.flatMap(lambda x:x.lower().split(" "))
words_2 = line_2wo_puct.flatMap(lambda x:x.lower().split(" "))
words_1 = line_1wo_puct.flatMap(lambda x:x.lower().split(" "))

total_length5 = words_5.count()
total_length4 = words_4.count()
total_length3 = words_3.count()
total_length2 = words_2.count()
total_length1 = words_1.count()

# printing the average length of comments for each rating
print("1 star rating: average length of comments ",round(total_length1/count_1,3))
print("2 star rating: average length of comments ",round(total_length2/count_2,3))
print("3 star rating: average length of comments ",round(total_length3/count_3,3))
print("4 star rating: average length of comments ",round(total_length4/count_4,3))
print("5 star rating: average length of comments ",round(total_length5/count_5,3))
