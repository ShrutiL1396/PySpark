<h1 align="center">Industry Analysis using Yelp Reviews</h1>

This project covers three main csv files: Business, User, and Review. It extracts some insights using these files individually and by combining them. The primary
focus is on building a model that can take any comment as input and determine whether that comment is made by an “Elite” Yelp user or by a “Non-Elite” Yelp user.

## Description
I have used three of the csv files provided in the Yelp dataset: **Business, User, and Review**. I have removed several columns thatare not a part of the analysis. 
Below are the columns that I have used from each of the csv files to find generalized insights. 

![Yelp Business Columns](https://github.com/ShrutiL1396/PySpark/blob/main/Yelp_Project/Yelp_Bizz.jpg) </br>

![Yelp Users Columns](https://github.com/ShrutiL1396/PySpark/blob/main/Yelp_Project/Yelp_rev.png) </br>

![Yelp Reviews Columns](https://github.com/ShrutiL1396/PySpark/blob/main/Yelp_Project/Yelp_uss.png) </br>

Yelp has two types of users **"Elite Users"** and **"Non-Elite Users"**. Yelp Elite users are a category of users who have well-written reviews, high quality tips, 
a detailed personal profile, an active voting and complimenting record, and a history of playing well with others. I have performed analysis around this idea of "Elite v/s Non-Elite".
After performing analysis on the Business file, User file and Review file, combined analysis has been performed on Business + User files, User + Review files and 
Business + User + Review files. Consequently I have also created a Machine Learning pipeline in the below format, whose aim is to determine whether a review left under
a business is written by an Elite user or a Non-Elite user on the basis of certain words or phrases which are prevalent in their reviews. Additionally I have also leveraged
GCP which runs the entire code and perfromed streaming operation such that, if a comment is written in real-time, the developed ML pipeline will determine whether it was written 
by an Elite or a Non-Elite user in real time.

![ML Pipeline](https://github.com/ShrutiL1396/PySpark/blob/main/Yelp_Project/ML_pipeline.png) </br>


## Prerequisites and Installation

- PySpark
- PySpark ML
- GCP Account with DataProc cluster
- Python
- Matplotlib
- Seaborn
- Plotly

## Contents
- [Yelp Project](https://github.com/ShrutiL1396/PySpark/tree/main/Yelp_Project) <br/>

- [Insight Analysis and ML Pipeline](https://github.com/ShrutiL1396/PySpark/blob/main/Yelp_Project/Scalable_Final_Project.py) </br>

- [Streaming operation with ML Pipeline](https://github.com/ShrutiL1396/PySpark/blob/main/Yelp_Project/Streaming.py) </br>


## Contact
Shruti Shivaji Lanke - <br/>
shrutilanke13@gmail.com or slanke1@student.gsu.edu <br/>
Project Link - <br/>
https://github.com/ShrutiL1396/PySpark/tree/main/Yelp_Project
