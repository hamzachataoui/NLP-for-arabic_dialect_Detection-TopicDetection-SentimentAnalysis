# Abstract
Natural Language Processing (NLP) is today a very active
field of research and innovation. Many applications need however big sets
of data for supervised learning, suitably labelled for the training purpose.
This includes applications for the Arabic language and its national dialects. However, such open access labeled data sets in Arabic and its
dialects are lacking in the Data Science ecosystem and this lack can be a
burden to innovation and research in this field. In this work, we present
an open data set of social data content in several Arabic dialects. This
data was collected from the Twitter social network and consists on +50K
twits in five (5) national dialects. Furthermore, this data was labeled for
several applications, namely dialect detection, topic detection and sentiment analysis. We publish this data as an open access data to encourage
innovation and encourage other works in the field of NLP for Arabic dialects and social media. A selection of models were built using this data
set and are presented in this paper along with their performances.
### Keywords: NLP · Open data · Supervised learning · Arab dialects.

#Data
The data was gathered by randomly scrapping tweets, from active users located
in a predefined set of Arab countries, namely : Algeria, Egypt, Lebanon, Tunisia
and Morocco. No limits were set for the date of the tweets nor for the exact
location in the country. We used Selenium Python library to automate the web
navigation and BeautifulSoup to scrap the tweets. The total number of tweets
in the data set is 49,306. The tweets distribution per country is given in table 1.

![image](https://user-images.githubusercontent.com/59541945/134767292-00c8a7be-9926-4c32-962d-8bcb3dacbf59.png) Table 1: Number of tweets per country




