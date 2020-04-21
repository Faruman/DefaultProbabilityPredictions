Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 22:22:05) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # -*- coding: utf-8 -*-
"""
The following file will extract the SEC fillings from the EDGAR webpage for all the companies that have submitted a file
to the SEC during a certain timeframe, 2002 to 2018 in our case. First, the full index links will be dowloaded, from recurring 
8-K, 10-Q or 10-K files among many other. Thereafter, only the direct link to 10-K files will be dowloaded as there 
are the one of interest. 
Based on the 10-K links, we will extract the full html files by withdrawing the tables and keeping only the full texts. 
We clean the dataset by withdrawing stop words, certain key words as "company" or the years of interests or by withdrawing
certain html tags. Finally, we count the numbers of positive and negative words in each 10-K file based on two different libraries: 
the one from Loughran and McDonald (2016) and the one from Bing Liu. Finally, a polarity score is extracted based on the 
number of positive and negative words from each 10-K file. 

ATTENTION: The given code below is extremely computational intensive and therefore should not be exectued diretly without
using an appropriate machine or slicing the workload in different steps. We decided to exectute the code below year by year,
 i.e. by changing line 93 and 95 to be the same year, ony by one, and it took us approximately a week to run the 17 years
with a CPU of 4GB and Virtual Memory up to 150GB. Therefore, we highly discourage to exectue the code below for an entire
timeframe on only one machine and advice to cut the process by year to be able to run it on several machines 
to be efficient. 
"""
#Import libraries needed
import os
import pandas as pd
import numpy as np
import requests
import sqlite3
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import nltk 
nltk.download('all')
import pickle
import nltk.data
from nltk.tokenize import RegexpTokenizer, sent_tokenize



#set working directory 
PATH = r"YOUR PATH HERE"
os.chdir(PATH)

# function to get the text from the html links via BeautifulSoup
def url_to_text(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    for table in soup.find_all('table'):
        table.decompose()
    text = soup.get_text()
    return text


#Function to clean the dataset
def clean_text_round2(text):
    # convert to lower case
    text = text.lower()
    text = re.sub(r'(\t|\v)', '', text)
    # remove \xa0 which is non-breaking space from ISO 8859-1
    text = re.sub(r'\xa0', ' ', text)
    # remove newline feeds (\n) following hyphens
    text = re.sub(r'(-+)\n{2,}', r'\1', text)
    # remove hyphens preceded and followed by a blank space
    text = re.sub(r'\s-\s', '', text)
    # replace 'and/or' with 'and or'
    text = re.sub(r'and/or', r'and or', text)
    # tow or more hypens, periods, or equal signs, possiblly followed by spaces are removed
    text = re.sub(r'[-|\.|=]{2,}\s*', r'', text)
    # all underscores are removed
    text = re.sub(r'_', '', text)
    # 3 or more spaces are replaced by a single space
    text = re.sub(r'\s{3,}', ' ', text)
    # three or more line feeds, possibly separated by spaces are replaced by two line feeds
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    # remove hyphens before a line feed
    text = re.sub(r'-+\n', '\n', text)
    # replace hyphens preceding a capitalized letter with a space
    text = re.sub(r'-+([A-Z].*)', r' \1', text)
    # remove capitalized or all capitals for the months
    text = re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)', '', text)
    # remove years
    text = re.sub(r'2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019', '', text)
    # remove words million and company 
    text = re.sub(r'million|company', '', text)  
    # remove line feeds
    text = re.sub('\n', ' ', text)
    #replace single line feed \n with single space
    text = re.sub(r'\n', ' ', text)
    return text


#Initiate the years
start_year = 2002
quarter = 4
end_year = 2018

years = list(range(start_year, end_year))
quarters = ['QTR1', 'QTR2', 'QTR3', 'QTR4']
history = [(y, q) for y in years for q in quarters]
#Look into the time frame wanted
for i in range(1, quarter + 1):
    history.append((end_year, 'QTR%d' % i))
urls = ['https://www.sec.gov/Archives/edgar/full-index/%d/%s/crawler.idx' % (x[0], x[1]) for x in history]
urls.sort()

#Pass via sqlite3 as is more efficient
con = sqlite3.connect('edgar_htm_idx.db')
cur = con.cursor()
cur.execute('DROP TABLE IF EXISTS idx')
cur.execute('CREATE TABLE idx (conm TEXT, type TEXT, cik TEXT, date TEXT, path TEXT)')

#Go get the required data from the EDGAR index file
for url in urls:
    lines = requests.get(url).text.splitlines()
    cikloc = lines[7].find('CIK')
    nameloc = lines[7].find('Company Name')
    typeloc = lines[7].find('Form Type')
    dateloc = lines[7].find('Date Filed')
    urlloc = lines[7].find('URL')
    records = [tuple([line[:typeloc].strip(), line[typeloc:cikloc].strip(), line[cikloc:dateloc].strip(),
                      line[dateloc:urlloc].strip(), line[urlloc:].strip()]) for line in lines[9:]]
    cur.executemany('INSERT INTO idx VALUES (?, ?, ?, ?, ?)', records)
    print(url, 'downloaded and wrote to SQLite')
 
con.commit()
con.close()

#After having passed by SQL, save it to a csv
engine2 = create_engine('sqlite:///edgar_htm_idx.db')
with engine2.connect() as conn, conn.begin():
    data_csv= pd.read_sql_table('idx', conn)
    
# Take only the 10-K files we are interested in 
files = ['10-K']
data_10K = data_csv[data_csv['type'].isin(files)]

#rearange the dataset 
data_10K = data_10K[['cik','conm','type','date','path']]

#save in csv
data_10K.to_csv('edgar_htm_idx.csv')

#Transform data into a list
firms = data_10K['conm']
firms =firms.values.tolist()

#Create a dataset only with the paths 
paths = data_10K.path

#Initialise the loop to extract the links 
final_link_appended=[]
n=1
for path in paths:
    first_link='None'
    url = path
    #Inspect element to find how the 10K files are stored - in <div> and then find type and the link right before that
    #this contains the actual 10 K file
    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')
    a_tags = soup.find_all('div')
    link1=path.split('/')
    link3="https://"+link1[1]+link1[2]+'/'
    first_link = soup.find_all(string='10-K')

    for i in first_link:
        if i=='10-K':
            end_link=i.find_previous('a')
            end1=str(end_link)
            pattern = re.compile(r'Archives.+\w.+htm"')
            match = pattern.search(end1)
            
            if match:
                formatted_link=format(match.group())
                formatted_link[:-1]
                final_link=link3+formatted_link[:-1]
                print(final_link)
                print(n)
                n=n+1
                final_link_appended.append(final_link) 
                
urls=final_link_appended

#Save all the links to the 10-K files into a csv file 
saved_url = pd.DataFrame(urls)
saved_url.to_csv('urls.csv')

#Get the texts files from the urls
urls = final_link_appended
texts = [url_to_text(url) for url in urls]

#save them into a csv file if needed
saved_texts = pd.DataFrame(texts)
saved_texts.to_csv('texts.csv')



stopWordsFile = PATH + r'\Dictionaries\StopWords_Generic.txt'
#Loading stop words dictionary for removing stop words
with open(stopWordsFile ,'r') as stop_words:
    stopWords = stop_words.read().lower()
stopWordList = stopWords.split('\n')
stopWordList[-1:] = []




#Tokenizeing module and filtering tokens using stop words list, removing punctuations
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens))
    return filtered_words

#Based on the dictionary of Loughran and McDonald (2016)
# Calculating positive score 
def positive_word_LM(text):
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in pos_dict_LM:
            numPosWords  += 1
    
    sumPos = numPosWords
    return sumPos

# Calculating Negative score
def negative_word_LM(text):
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in neg_dict_LM:
            numNegWords -=1
    sumNeg = numNegWords 
    sumNeg = sumNeg * -1
    return sumNeg

#Based on the dictionary of Bing Liu 
# Calculating positive score 
def positive_word_B(text):
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in pos_dict_B:
            numPosWords  += 1
    
    sumPos = numPosWords
    return sumPos

# Calculating Negative score
def negative_word_B(text):
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in neg_dict_B:
            numNegWords -=1
    sumNeg = numNegWords 
    sumNeg = sumNeg * -1
    return sumNeg

# Calculating polarity score
def polarity_score(positiveScore, negativeScore):
    pol_score = (positiveScore - negativeScore) / ((positiveScore + negativeScore) + 0.000001)
    return pol_score


# Function to count the words
def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)

# Calculating Average sentence length 
# It will calculated using formula --- Average Sentence Length = the number of words / the number of sentences
     
def average_sentence_length(text):
    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length= average_sent
    
    return round(average_sent_length)



#Import the negative and positive dictionaries 
##### Words from Loughran and McDonald
# negative 
neg_dict_LM = ""
neg_dict_LM = pd.read_csv(PATH + r"\Dictionaries\lm_negative.csv",encoding = 'ISO-8859-1', names=["lm_negative"])["lm_negative"].values.tolist()
neg_dict_LM = str(neg_dict_LM)
neg_dict_LM = neg_dict_LM.lower()

# positive
pos_dict_LM = ""
pos_dict_LM = pd.read_csv(PATH + r"\Dictionaries\lm_positive.csv", encoding = 'ISO-8859-1', names=["lm_positive"])["lm_positive"].values.tolist()
pos_dict_LM = str(pos_dict_LM)
pos_dict_LM = pos_dict_LM.lower()

##### Words from Bing Liu
# negative
neg_dict_B = ""
neg_dict_B = pd.read_csv(PATH + r"\Dictionaries\bl_negative.csv", encoding = 'ISO-8859-1', names=["bl_negative"])["bl_negative"].values.tolist()
neg_dict_B = str(neg_dict_B)
neg_dict_B = neg_dict_B.lower()


# positive
pos_dict_B = ""
pos_dict_B = pd.read_csv(PATH + r"\Dictionaries\bl_positive.csv",encoding = 'ISO-8859-1', names=["bl_positive"])["bl_positive"].values.tolist()
pos_dict_B = str(pos_dict_B)
pos_dict_B = pos_dict_B.lower()

# If the process of collection and textual analysis were different, here dowload the urls 
#urls = pd.read_csv(PATH + r"\urls.csv", index_col = 0, names=["url"])["url"].values.tolist()[1:]

# Retrive all the cik numbers from the urls 
ciks = []
for url in urls: 
    cik = []
    path = url 
    link1 = path.split('/')
    cik = link1[6]
    ciks.append(cik)
print("ciks fetched")


# Pickle the data to save them on the computer
for idx, val in enumerate(ciks):
    with open(PATH + r"\pickle" + val+"val" + ".pkl", "wb") as f:
        if idx != len(ciks)-1:
            pickle.dump(texts[idx], f)
        else: 
            pickle.dump(texts[idx-1], f)

#Open the pickle files previously saved on the computer                
data = {}
for _, val in enumerate(ciks):
    
    with open(PATH + r"\pickle" + val +"val"+ ".pkl", "rb") as f:
        data[val] = pickle.load(f)
        
# Change to a pd.DataFrame
df = pd.DataFrame.from_dict(data, orient = "index")
df1 = pd.DataFrame(df[0].apply(clean_text_round2))
    
# Create the variables for the textual analysis 
df1['word_count'] = df1.iloc[:,0].apply(total_word_count)
df1['positive_LM'] = df1.iloc[:,0].apply(positive_word_LM)
df1['negative_LM'] = df1.iloc[:,0].apply(negative_word_LM)
df1['polarity_LM'] = np.vectorize(polarity_score)(df1['positive_LM'],df1['negative_LM'])
df1['positive_B'] = df1.iloc[:,0].apply(positive_word_B)
df1['negative_B'] = df1.iloc[:,0].apply(negative_word_B)
df1['polarity_B'] = np.vectorize(polarity_score)(df1['positive_B'],df1['negative_B'])
df1['average_sentence_lenght'] = df1.iloc[:,0].apply(average_sentence_length)

df1.to_csv(PATH + r"\results\textual_analyis.csv")
print("done")
