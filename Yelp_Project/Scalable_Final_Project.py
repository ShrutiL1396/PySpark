# pip install plotly
# pip install kaleido

"""## Importing necessary modeules """
from pyspark.sql import SparkSession
from pyspark.sql.functions import ltrim,rtrim,trim
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import round
from pyspark.sql.functions import col, to_date
import datetime
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.sql.functions import year, month, dayofmonth
import math
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import *
from pyspark.sql.functions import split,explode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("dark")

"""## Setting spark configuration to run on cloud """
conf = SparkConf().setMaster("local[*]").set("spark.executer.memory", "4g")
sc = SparkContext(conf=conf)
spark = SparkSession(sc).builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import explode, udf
from pyspark.sql.types import *
from pyspark.sql.functions import col
from pyspark.ml.feature import * 
import re
import string
import pyspark.sql.functions as func

import nltk
import re
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords_nltk = set(nltk.corpus.stopwords.words('english'))

"""## Reading the business csv """
business = spark.read.format("csv").option("header","true").option("delimiter",",").option("quote", "\"").option("escape", "\"").load("./yelp_business.csv")

business.show()

business.printSchema()

business_filtered = business.select('business_id','name','city','state','stars','review_count','is_open','categories')

business_filtered.show()

"""## Selecting and grouping by locations """
locations = business.select('business_id','city')
most_reviewed_city = locations.groupby('city').count()
cities = most_reviewed_city.sort('count',ascending=False)

pdf = cities.toPandas()

pdf_top10 = pdf.head(10)

pdf_top10

"""## Exploratory Data Analysis"""

"""##  Plotting top cities with max business count """
city = pdf_top10['city']
counts = pdf_top10['count']
fig = plt.figure(figsize=(10,6))
plt.bar(city, counts, color ='orange',
        width = 0.4)
plt.xlabel('City', fontsize=15)
plt.xticks(rotation=45,fontsize=11)
plt.ylabel('No. of Businesses', fontsize=15)
plt.title('Top 10 Cities with Maximum Businesses', fontsize=15)
plt.savefig('fig1.png')

open = business.select('business_id','is_open')
open = open.groupby('is_open').count()

opendf = open.toPandas()

opendf

"""##  Plotting open vs closed businesses """
is_open = opendf['is_open']
counts = opendf['count']
fig = plt.figure(figsize=(10,6))
plt.pie(counts,labels=is_open,autopct='%1.1f%%')
plt.legend()
plt.title('Share of Open vs Closed Businesses', fontsize=15)
plt.savefig('fig2.png')


category = business.select('categories')
individual_category = category.select(explode(split('categories', ';')).alias('category'))
grouped_category = individual_category.groupby('category').count()
top_category = grouped_category.sort('count',ascending=False)
top_category.show(10,truncate=False)

categorydf = top_category.toPandas()

categorydf_top10 = categorydf.head(10)

categorydf_top10

"""##  Plotting categories vs the count of businesses """
category = categorydf_top10['category']
counts = categorydf_top10['count']
fig = plt.figure(figsize=(10,6))
plt.bar(category, counts, color ='red')
plt.xlabel('Category', fontsize=15)
plt.xticks(rotation = 45,fontsize=11)
plt.ylabel('Number of Businesses Listing This Category', fontsize=15)
plt.title('Most Famous Categories from Yelp', fontsize=15)
plt.savefig('fig3.png')

stars = business.select('stars','business_id')
stars_group = stars.groupby('stars').count()
stars_sorted = stars_group.sort('stars',ascending=True)

starsdf = stars_sorted.toPandas()

"""##  Plotting max star rated count of businesses """
stars = starsdf['stars']
counts = starsdf['count']
fig = plt.figure(figsize=(10,6))
plt.bar(stars, counts, color=['C0', 'C1', 'C2', 'C3', 'C4','C5','C6','C7','C8'])
plt.xlabel('Stars', fontsize=15)
plt.xticks(fontsize=11)
plt.ylabel('No. of Businesses', fontsize=15)
plt.title('Rating Distribution of Businesses', fontsize=15)
plt.savefig('fig4.png')

business_pop = business.select('business_id','name','review_count','stars')

business_popular = business_pop.withColumn("Stars_Rating", F.col("stars")*F.col("review_count"))

most_popular = business_popular.sort('Stars_Rating',ascending=False)

top_10_pop = most_popular.limit(10)

top10df = top_10_pop.toPandas()

name = top10df['name']
popularity = top10df['Stars_Rating']

"""##  Plotting most popular businesses """
fig = plt.figure(figsize=(10,6))
plt.bar(name, popularity, color ='indigo')
plt.xlabel('Business Name', fontsize=15)
plt.xticks(rotation = 90,fontsize=11)
plt.ylabel('Popularity', fontsize=15)
plt.title('Most Popular Businesses', fontsize=15)
plt.savefig('fig5.png')

"""## Review Data Set"""

review = spark.read.format("csv").option("header","true").option("delimiter",",").option("quote", "\"").option("escape", "\"").load("./yelp_review.csv")

review.printSchema()

review = review.filter(review.stars.isin('5', '4', '3', '2', '1'))
review = review.filter("text is not null")
review = review.filter("date is not null")
review = review.withColumn("stars", review['stars'].cast("int"))
review = review.filter("stars is not null")
review.show()

review_per_star = review.select('review_id', 'stars')
count_per_star = review_per_star.groupby('stars').count()
cnt_str = count_per_star.sort('count', ascending = False)
pdf_stars = cnt_str.toPandas()
pdf_stars

"""## Distribution is Comments per Rating """
plt.figure(figsize=(10,6))
pdf_stars.plot(kind='bar',x='stars',y='count', figsize=(8,6),color='orange')
plt.xlabel('Ratings', fontsize=15)
plt.xticks(rotation = 0, fontsize = 11)
plt.ylabel('No. of Reviews', fontsize=15)
plt.title('Distribution is Comments per Rating', fontsize=15)
plt.savefig('fig6.png')

word_count = review.select( \
    col("stars"), \
    trim(col("text")).alias("text"), \
    )
review.select('stars', 'text')
word_count = word_count.withColumn('length', f.size(f.split(f.col('text'), ' ')))
word_count.show()

avg_word_cnt = word_count.groupBy("stars").avg("length")
avg_word_cnt = avg_word_cnt.withColumnRenamed("avg(length)","Avg_Comment_Length")
avg_word_cnt = avg_word_cnt.withColumn("Avg_Comment_Length", round(col("Avg_Comment_Length"), 2))
cnt_wrds = avg_word_cnt.sort('Avg_Comment_Length', ascending = False)
pdf = cnt_wrds.toPandas()
pdf.head()

"""##  Average Length of Reviews per Rating """
plt.figure(figsize=(10,6))
pdf.plot(kind='bar',x='stars',y='Avg_Comment_Length', figsize=(8,6), color='purple')
plt.xlabel('Ratings', fontsize=15)
plt.xticks(rotation = 0, fontsize=11)
plt.ylabel('Average Length of Reviews', fontsize=15)
plt.title('Average Length of Reviews per Rating', fontsize=15)
plt.savefig('fig7.png')

from pyspark.sql.functions import *

month_comment = review.select(col("text"), col("date"), to_date(col("date"),"yyyy-MM-dd"))
month_comment = month_comment.drop(col("date"))
month_comment = month_comment.withColumnRenamed("to_date(date, yyyy-MM-dd)","date")
month_comment = month_comment.filter("text is not null")
month_comment = month_comment.filter("date is not null")
month_comment = month_comment.select(col("text"), date_format(col("date"), "MMMM").alias("month"))
month_comment.show()

month_comment.printSchema()

mnth_com = month_comment.groupby('month').count()
pdf_mon = mnth_com.toPandas()
pdf_mon

month = pdf_mon['month']
rev_mon_cnt = pdf_mon['count']

"""## Monthly Count of Comments over the Years """
fig = plt.figure(figsize=(10,6))
plt.bar(month, rev_mon_cnt, color ='blue', width = 0.4)
plt.xlabel('Monthly Distribution of Comments', fontsize=15)
plt.xticks(rotation = 45, fontsize=11)
plt.ylabel('No. of Reviews', fontsize=15)
plt.title('Monthly Count of Comments over the Years', fontsize=15)
plt.savefig('fig8.png')

review_filtered = review.select('text', 'stars')

"""## Ranking of Words in Reviews: """

lines = review_filtered.rdd.map(tuple)

lines.persist()

stopword_list = set(stopwords_nltk)
def ProcessedText(text):
	tokens = nltk.word_tokenize(text)
	punct_rem = [word for word in tokens if word.isalpha()]
	remove_stopwords = [word for word in punct_rem if not word in stopword_list]
	return remove_stopwords

line_5star = lines.filter(lambda x: x[1] == 5)
line_4star = lines.filter(lambda x: x[1] == 4)
line_3star = lines.filter(lambda x: x[1] == 3)
line_2star = lines.filter(lambda x: x[1] == 2)
line_1star = lines.filter(lambda x: x[1] == 1)

words_5star = line_5star.flatMap(lambda x: ProcessedText(str(x).lower()))
words_4star = line_4star.flatMap(lambda x: ProcessedText(str(x).lower()))
words_3star = line_3star.flatMap(lambda x: ProcessedText(str(x).lower()))
words_2star = line_2star.flatMap(lambda x: ProcessedText(str(x).lower()))
words_1star = line_1star.flatMap(lambda x: ProcessedText(str(x).lower()))

words_5_freq = words_5star.map(lambda x:(x,1))
words_4_freq = words_4star.map(lambda x:(x,1))
words_3_freq = words_3star.map(lambda x:(x,1))
words_2_freq = words_2star.map(lambda x:(x,1))
words_1_freq = words_1star.map(lambda x:(x,1))

res_5 = words_5_freq.reduceByKey(lambda x,y:x+y)
res_4 = words_4_freq.reduceByKey(lambda x,y:x+y)
res_3 = words_3_freq.reduceByKey(lambda x,y:x+y)
res_2 = words_2_freq.reduceByKey(lambda x,y:x+y)
res_1 = words_1_freq.reduceByKey(lambda x,y:x+y)

sort_5 = (sorted(res_5.collect(),key = lambda x:x[1],reverse=True)[:10])
sort_4 = (sorted(res_4.collect(),key = lambda x:x[1],reverse=True)[:10])
sort_3 = (sorted(res_3.collect(),key = lambda x:x[1],reverse=True)[:10])
sort_2 = (sorted(res_2.collect(),key = lambda x:x[1],reverse=True)[:10])
sort_1 = (sorted(res_1.collect(),key = lambda x:x[1],reverse=True)[:10])

sort_5_b = sc.parallelize(sort_5)
sort_5_words = sort_5_b.map(lambda x: x[0])
sort_4_b = sc.parallelize(sort_4)
sort_4_words = sort_4_b.map(lambda x: x[0])
sort_3_b = sc.parallelize(sort_3)
sort_3_words = sort_3_b.map(lambda x: x[0])
sort_2_b = sc.parallelize(sort_2)
sort_2_words = sort_2_b.map(lambda x: x[0])
sort_1_b = sc.parallelize(sort_1)
sort_1_words = sort_1_b.map(lambda x: x[0])

print("1 star rating: ",sort_1_words.collect())
print("2 star rating: ",sort_2_words.collect())
print("3 star rating: ",sort_3_words.collect())
print("4 star rating: ",sort_4_words.collect())
print("5 star rating: ",sort_5_words.collect())

"""## Calculate Average Length of Reviews"""

lines = review_filtered.rdd.map(tuple)

lines.persist()

def ProcessedText(text):
    return re.sub('\W+', ' ', str(text))

line_5star = lines.filter(lambda x: x[1] == 5)
line_4star = lines.filter(lambda x: x[1] == 4)
line_3star = lines.filter(lambda x: x[1] == 3)
line_2star = lines.filter(lambda x: x[1] == 2)
line_1star = lines.filter(lambda x: x[1] == 1)

count_5 = line_5star.count()
count_4 = line_4star.count()
count_3 = line_3star.count()
count_2 = line_2star.count()
count_1 = line_1star.count()

line_5_punct_rem = line_5star.map(lambda x:ProcessedText(x))
line_4_punct_rem = line_4star.map(lambda x:ProcessedText(x))
line_3_punct_rem = line_3star.map(lambda x:ProcessedText(x))
line_2_punct_rem = line_2star.map(lambda x:ProcessedText(x))
line_1_punct_rem = line_1star.map(lambda x:ProcessedText(x))

words_5 = line_5_punct_rem.flatMap(lambda x:x.lower().split(" "))
words_4 = line_4_punct_rem.flatMap(lambda x:x.lower().split(" "))
words_3 = line_3_punct_rem.flatMap(lambda x:x.lower().split(" "))
words_2 = line_2_punct_rem.flatMap(lambda x:x.lower().split(" "))
words_1 = line_1_punct_rem.flatMap(lambda x:x.lower().split(" "))

words_5.take(10)

total_length_5_star = words_5.count()
total_length_4_star = words_4.count()
total_length_3_star = words_3.count()
total_length_2_star = words_2.count()
total_length_1_star = words_1.count()

avg_count_5 = total_length_5_star/count_5
avg_count_4 = total_length_4_star/count_4
avg_count_3 = total_length_3_star/count_3
avg_count_2 = total_length_2_star/count_2
avg_count_1 = total_length_1_star/count_1

avg_count_1

print("5 star rating: average length of comments ", avg_count_5)
print("4 star rating: average length of comments ", avg_count_4)
print("3 star rating: average length of comments ", avg_count_3)
print("2 star rating: average length of comments ", avg_count_2)
print("1 star rating: average length of comments ", avg_count_1)

"""## Review Visualization:"""

review_per_star = review.select('review_id', 'stars')
count_per_star = review_per_star.groupby('stars').count()
cnt_str = count_per_star.sort('count', ascending = False)
pdf_stars = cnt_str.toPandas()
pdf_stars

plt.figure(figsize=(10,6))
pdf_stars.plot(kind='bar',x='stars',y='count', figsize=(8,6),color='orange')
plt.xlabel('Ratings', fontsize=15)
plt.xticks(rotation = 0, fontsize = 11)
plt.ylabel('No. of Reviews', fontsize=15)
plt.title('Distribution is Comments per Rating', fontsize=15)
plt.savefig("fig17.png")

word_count = review.select( \
    col("stars"), \
    trim(col("text")).alias("text"), \
    )
review.select('stars', 'text')
word_count = word_count.withColumn('length', f.size(f.split(f.col('text'), ' ')))
word_count.show()

avg_word_cnt = word_count.groupBy("stars").avg("length")
avg_word_cnt = avg_word_cnt.withColumnRenamed("avg(length)","Avg_Comment_Length")
avg_word_cnt = avg_word_cnt.withColumn("Avg_Comment_Length", round(col("Avg_Comment_Length"), 2))
cnt_wrds = avg_word_cnt.sort('Avg_Comment_Length', ascending = False)
pdf = cnt_wrds.toPandas()
pdf.head()

plt.figure(figsize=(10,6))
pdf.plot(kind='bar',x='stars',y='Avg_Comment_Length', figsize=(8,6), color='purple')
plt.xlabel('Ratings', fontsize=15)
plt.xticks(rotation = 0, fontsize=11)
plt.ylabel('Average Length of Reviews', fontsize=15)
plt.title('Average Length of Reviews per Rating', fontsize=15)
plt.savefig("fig18.png")

from pyspark.sql.functions import *

month_comment = review.select(col("text"), col("date"), to_date(col("date"),"yyyy-MM-dd"))
month_comment = month_comment.drop(col("date"))
month_comment = month_comment.withColumnRenamed("to_date(date, yyyy-MM-dd)","date")
month_comment = month_comment.filter("text is not null")
month_comment = month_comment.filter("date is not null")
month_comment = month_comment.select(col("text"), date_format(col("date"), "MMMM").alias("month"))
month_comment.show()

month_comment.printSchema()

mnth_com = month_comment.groupby('month').count()
pdf_mon = mnth_com.toPandas()
pdf_mon

month = pdf_mon['month']
rev_mon_cnt = pdf_mon['count']
fig = plt.figure(figsize=(10,6))
plt.bar(month, rev_mon_cnt, color ='blue', width = 0.4)
plt.xlabel('Monthly Distribution of Comments', fontsize=15)
plt.xticks(rotation = 45, fontsize=11)
plt.ylabel('No. of Reviews', fontsize=15)
plt.title('Monthly Count of Comments over the Years', fontsize=15)
plt.savefig("fig19.png")

user = spark.read.option("delimiter",",").option("header","true").csv('./yelp_user.csv')

user.printSchema()

user_filtered = user.select('user_id','name','review_count','yelping_since','friends','useful','funny','cool','elite','average_stars')

user_filtered = user_filtered.withColumn("average_stars",user_filtered['average_stars'].cast("double"))

"""**Rounding off the values for "Average_Stars" column**"""

user_filtered = user_filtered.withColumn("average_stars",func.round(user_filtered['average_stars']))

"""**Checking for null records in the Dataframe**


"""

user_filtered.select([func.count(func.when(func.isnull(c), c)).alias(c) for c in user_filtered.columns]).show()

"""**Dropping null records**"""

user_filtered_nna = user_filtered.dropna()

user_filtered_nna.select([func.count(func.when(func.isnull(c), c)).alias(c) for c in user_filtered.columns]).show()

user_filtered_nna.count()

user_filtered_nna.show()



"""**Total number of Elite users**"""

print('Total number of Elite users:', \
      user_filtered_nna.filter(~user_filtered_nna["elite"].isin('None')).count())

"""**Total number of Non-Elite users**"""

print('Total number of Non-Elite users:', \
      user_filtered_nna.filter(user_filtered_nna.elite.isin('None')).count())

"""**Filter our Elite users**"""

elite_users = user_filtered_nna.filter(~user_filtered_nna["elite"].isin('None'))

res = elite_users.groupby("average_stars").count()

el_res = sorted(res.collect())

"""**Rating Distribution for Elite Users**"""

for i in el_res:
  print("Number of "+str(i[0])+" star ratings given by elite users:- "+(str(i[1])))

pdf = res.toPandas()

pdf.rename(columns={'count':'Count'},inplace=True)
pdf.sort_values(by='Count',ascending=True,inplace=True)

pdf['Count'] = pdf['Count'].astype('int')

size = pdf["Count"].apply(lambda x: math.log(x))
plt.figure(figsize=(8,5))
sns.scatterplot(data=pdf,x="average_stars",y="Count", s=size*40, hue='average_stars', \
                palette = 'crest',linewidth=2)
plt.text(2.8, 4400, "8,034 reviews", horizontalalignment='left', size='medium', color='black')
plt.text(1.9, 1500, "11 reviews", horizontalalignment='left', size='medium', color='black')
plt.text(4.6, 3500, "1,469 reviews", horizontalalignment='left', size='medium', color='black')
plt.text(3.8, 47000, "51,303 reviews", horizontalalignment='left', size='medium', color='black')
plt.title('Distribution of Average Rating for Elite Users')
plt.savefig('fig13.png')

"""**Representing Non-Elite Users**"""

non_elite = user_filtered_nna.filter(user_filtered_nna.elite.isin('None'))

res_ne = non_elite.groupby("average_stars").count()

res_ne_s = sorted(res_ne.collect())

"""**Rating Distribution for Non-Elite Users**"""

for i in res_ne_s:
  print("Number of "+str(i[0])+" star ratings given by non-elite users:- "+(str(i[1])))

pdf_2 = res_ne.toPandas()

pdf_2.rename(columns={'count':'Count'},inplace=True)
pdf_2.sort_values(by='average_stars',ascending=True,inplace=True)

pdf_2['Count'] = pdf_2['Count'].astype('int')

size = pdf_2["Count"].apply(lambda x: math.log(x))
plt.figure(figsize=(9,5))
sns.scatterplot(data=pdf_2,x="average_stars",y="Count", s=size*40, hue='average_stars', \
                palette = 'flare',linewidth=2)
plt.text(0.9, 110000, "93,885 reviews", horizontalalignment='left', size='medium', color='black')
plt.text(1.8, 105000, "86,181 reviews", horizontalalignment='left', size='medium', color='black')
plt.text(2.75, 280000, "257,517 reviews", horizontalalignment='left', size='medium', color='black')
plt.text(3.75, 419000, "448,104 reviews", horizontalalignment='left', size='medium', color='black')
plt.text(4.45, 350000, "379,101 reviews", horizontalalignment='left', size='medium', color='black')
plt.title('Distribution of Average Rating for Non-Elite Users')
plt.savefig('fig14.png')

elite_users = elite_users.withColumn("useful",elite_users['useful'].cast("int"))

elite_users = elite_users.withColumn("funny",elite_users['funny'].cast("int"))

elite_users = elite_users.withColumn("cool",elite_users['cool'].cast("int"))

fil_elu_u = elite_users.filter(~elite_users.useful.isin('0') & ~elite_users.cool.isin('0') & ~elite_users.funny.isin('0') )

use_avg = fil_elu_u.agg({'useful': 'avg','funny':'avg','cool':'avg'})
use_avg = use_avg.withColumn("Useful",F.round(use_avg['avg(useful)'])) #.show()
use_avg = use_avg.withColumn("Cool",F.round(use_avg['avg(cool)']))
use_avg = use_avg.withColumn("Funny",F.round(use_avg['avg(funny)']))

use_avg = use_avg.select('Useful','Cool','Funny')

df2 = use_avg.toPandas()
df2 = df2.transpose().reset_index()
df2.rename(columns={'index':'User Rating',0:'Average'},inplace=True)
df2['Average'] = df2['Average'].astype('int')

plt.figure(figsize=(10,8))
ax = sns.barplot(data=df2,x="User Rating",y="Average",palette='Blues_d')
plt.title('Distribution of Average User Rating for Elite',fontsize=16)
patches = ax.patches
avg = df2['Average']

for i in range(len(patches)):
    x = patches[i].get_x() + patches[i].get_width()/2
    y = patches[i].get_height()+.05
    ax.annotate('{}'.format(avg[i]), (x, y), ha='center', fontsize=13)

plt.show()
plt.savefig('fig15.png')

non_elite = non_elite.withColumn("useful",non_elite['useful'].cast("int"))
non_elite = non_elite.withColumn("funny",non_elite['funny'].cast("int"))
non_elite = non_elite.withColumn("cool",non_elite['cool'].cast("int"))

filu_non_elite = non_elite.filter(~non_elite.useful.isin('0') & ~non_elite.cool.isin('0') & ~non_elite.funny.isin('0') )

use_neavg = filu_non_elite.agg({'useful': 'avg','funny':'avg','cool':'avg'})
use_neavg = use_neavg.withColumn("Useful",F.round(use_neavg['avg(useful)'])) #.show()
use_neavg = use_neavg.withColumn("Cool",F.round(use_neavg['avg(cool)']))
use_neavg = use_neavg.withColumn("Funny",F.round(use_neavg['avg(funny)']))

use_neavg = use_neavg.select('Useful','Cool','Funny')

df3 = use_neavg.toPandas()
df3 = df3.transpose().reset_index()
df3.rename(columns={'index':'User Rating',0:'Average'},inplace=True)
df3['Average'] = df3['Average'].astype('int')

plt.figure(figsize=(10,8))
ax = sns.barplot(data=df3,x="User Rating",y="Average",palette='Greens_r')
plt.title('Distribution of Average User Rating for Non-Elite',fontsize=16)
patches = ax.patches
avg = df3['Average']

for i in range(len(patches)):
    x = patches[i].get_x() + patches[i].get_width()/2
    y = patches[i].get_height()+.05
    ax.annotate('{}'.format(avg[i]), (x, y), ha='center', fontsize=13)

plt.show()
plt.savefig('fig16.png')

reviews = spark.read.format("csv").option("header","true").option("delimiter",",").option("quote", "\"").option("escape", "\"").load("./yelp_review.csv")

reviews = reviews.filter("review_id is not null")
reviews = reviews.filter("user_id is not null")
reviews = reviews.filter("business_id is not null")
reviews = reviews.filter("date is not null")
reviews = reviews.filter("text is not null")

reviews = reviews.filter(reviews.stars.isin('5', '4', '3', '2', '1'))

reviews.count()

all_users = user_filtered_nna.select('user_id','elite')
ref_rev = reviews.select('user_id','text','business_id','stars')
all_user_reviews = all_users.join(ref_rev,'user_id','inner')

all_user_reviews.show()

from pyspark.sql.functions import when, col

conditions = when(col("elite").isin('None'), 0).otherwise(1)
all_user_reviews = all_user_reviews.withColumn("Is_Elite", conditions)

all_user_reviews.show()

"""**Top 20 words**"""

test_df_fc = all_user_reviews.select('text','Is_Elite')

el_wc = test_df_fc.filter(test_df_fc.Is_Elite.isin(1))
nel_wc = test_df_fc.filter(test_df_fc.Is_Elite.isin(0))

el_wc = el_wc.select('text','Is_Elite')
nel_wc = nel_wc.select('text','Is_Elite')

df_clean = el_wc.select('text','Is_Elite',trim(lower(regexp_replace('text', "[^a-zA-Z\\s]", ""))).alias('clean_text'))
# # Tokenize text
tokenizer = Tokenizer(inputCol='clean_text', outputCol='words_token')
df_words_token = tokenizer.transform(df_clean).select('text','is_Elite', 'words_token')

# # Remove stop words
stopwordList = ["one","get","go","ive","also","came","back","im","u","us","got","dont","even"] 
stopwordList.extend(StopWordsRemover().getStopWords())
remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean',stopWords=stopwordList)
df_words_no_stopw = remover.transform(df_words_token).select('text','is_Elite', 'words_clean')

result = df_words_no_stopw.withColumn('word', F.explode(F.col('words_clean'))) \
  .groupBy('word') \
  .count().sort('count', ascending=False)

#df_clean.withColumnRenamed('trim(lower(regexp_replace(text, [^a-zA-Z\s], , 1)))','clean_text')
result.show(20)

df_clean2 = nel_wc.select('text','Is_Elite',trim(lower(regexp_replace('text', "[^a-zA-Z\\s]", ""))).alias('clean_text'))
# # Tokenize text
tokenizer2 = Tokenizer(inputCol='clean_text', outputCol='words_token')
df_words_token2 = tokenizer.transform(df_clean2).select('text','is_Elite', 'words_token')

# # Remove stop words
stopwordList = ["one","get","go","ive","also","came","back","im","u","us","got","dont","even","went"] 
stopwordList.extend(StopWordsRemover().getStopWords())
remover2 = StopWordsRemover(inputCol='words_token', outputCol='words_clean',stopWords=stopwordList)
df_words_no_stopw2 = remover2.transform(df_words_token2).select('text','is_Elite', 'words_clean')

result2 = df_words_no_stopw2.withColumn('word', F.explode(F.col('words_clean'))) \
  .groupBy('word') \
  .count().sort('count', ascending=False)

result2.show(20)

ref_bussiness = business.select('business_id','name','state','categories')
buss_usr_rev = all_user_reviews.join(ref_bussiness, 'business_id','inner')

buss_usr_rev.show()

"""**Elite users tend to review which kind of businesses the most**"""

from pyspark.sql.functions import split,explode
eli_buss = buss_usr_rev.filter(buss_usr_rev.Is_Elite.isin(1))
category = eli_buss.select('categories')
individual_category = category.select(explode(split('categories', ';')).alias('category'))
grouped_category = individual_category.groupby('category').count()
top_category = grouped_category.sort('count',ascending=False)
top_category.show(10,truncate=False)

df_5 = top_category.toPandas()

df_5 = df_5.head(10)

df_5

import plotly.express as px
fig = px.pie(df_5, values='count', names='category', title='Businesses popular among Elite Users',\
             color_discrete_sequence=px.colors.sequential.matter)
fig.show()
fig.write_image('fig9.png',engine="kaleido")

"""**Non-Elite users tend to review which kind of businesses the most**"""

from pyspark.sql.functions import split,explode
no_eli_buss = buss_usr_rev.filter(buss_usr_rev.Is_Elite.isin(0))
category2 = no_eli_buss.select('categories')
individual_category2 = category2.select(explode(split('categories', ';')).alias('category'))
grouped_category2 = individual_category2.groupby('category').count()
top_category2 = grouped_category2.sort('count',ascending=False)
top_category2.show(10,truncate=False)

df_4 = top_category2.toPandas()

df_4 = df_4.head(10)

df_4

fig = px.pie(df_4, values='count', names='category', title='Businesses popular among Non-Elite Users',\
             color_discrete_sequence=px.colors.sequential.haline)
fig.show()
fig.write_image('fig10.png',engine="kaleido")

top_category2

"""**Top Businesses Reviewed by Elite Users**"""

from pyspark.sql.functions import split,explode
eli_name = buss_usr_rev.filter(buss_usr_rev.Is_Elite.isin(1))
name_buss = eli_name.select('name')
grouped_category3 = name_buss.groupby('name').count()
top_buss = grouped_category3.sort('count',ascending=False)
top_buss.show(10,truncate=False)

df_6 = top_buss.toPandas()

df_6 = df_6.head(10)

df_6['count'] = df_6['count'].astype('float')

df_6.info()

df_6.plot(kind='barh',x='name',y='count', figsize=(8,6),color='navy')
plt.title("Business outlets popular among Elite Users",fontsize=18)
plt.xlabel('Number of Reviews')
plt.ylabel('Name of the Business')
for index,row in df_6.iterrows():
  row = (row['count'])
  label = format(int(row),',')
  plt.annotate(label,xy=(row - 600,index - 0.1),color='white',fontweight='bold')
plt.savefig('fig11.png')

"""**Top Businesses Reviewed by Non-Elite Users**"""

from pyspark.sql.functions import split,explode
no_eli_name = buss_usr_rev.filter(buss_usr_rev.Is_Elite.isin(0))
no_name_buss = no_eli_name.select('name')
grouped_category4 = no_name_buss.groupby('name').count()
top_buss2 = grouped_category4.sort('count',ascending=False)
top_buss2.show(10,truncate=False)

df_7 = top_buss2.toPandas()

df_7 = df_7.head(10)

df_7['count'] = df_7['count'].astype('float')

df_7.plot(kind='barh',x='name',y='count', figsize=(8,6),color='teal')
plt.title("Business outlets popular among Non-Elite Users",fontsize=18)
plt.xlabel('Number of Reviews')
plt.ylabel('Name of the Business')
for index,row in df_7.iterrows():
  row = (row['count'])
  label = format(int(row),',')
  plt.annotate(label,xy=(row - 1200,index - 0.1),color='white',fontweight='bold')
plt.savefig('fig12.png')

"""## Classification Model """

all_user_text = all_user_reviews.select('text','Is_Elite')

all_user_text.count()

from pyspark.ml.feature import Tokenizer, HashingTF, IDF 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

import nltk
import re

nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords_nltk = set(nltk.corpus.stopwords.words('english'))

nltk.download('punkt')

all_user_text.printSchema()

all_user_text.show(2)

#2 ML pipeline

from pyspark.ml.feature import StopWordsRemover

tokenizer = Tokenizer(inputCol = "text", outputCol = "words") 
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
TF = HashingTF(inputCol = tokenizer.getOutputCol(), outputCol = "tfFeatures")
idf = IDF(inputCol = "tfFeatures",outputCol = "features")
lr = LogisticRegression(maxIter = 10, regParam = 0.001)
pipeline = Pipeline(stages = [tokenizer,remover,TF,idf,lr])

all_user_text = all_user_text.withColumnRenamed("Is_Elite","label")

train = all_user_text.sampleBy("label", fractions={0: 0.9, 1: 0.9}, seed=10)

test = all_user_text.subtract(train)

train.show()

train.groupBy('label').count().show()

model = pipeline.fit(train)

""" ## For saving model, uncomment below code """
#model.save("./elite_model.model")

prediction = model.transform(test)

prediction.show()

acc = prediction.select("label","prediction")

conditions = when(col("label")==col("prediction"), 1).otherwise(0)
acc2 = acc.withColumn("Valid", conditions)

acc3 = acc2.groupBy("Valid").count()

acc3df = acc3.toPandas()

print("Accuracy of the model is: ", (acc3df.iloc[0,1]/(acc3df.iloc[1,1] + acc3df.iloc[0,1]))*100)
