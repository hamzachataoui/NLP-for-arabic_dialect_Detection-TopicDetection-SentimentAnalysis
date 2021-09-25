# Abstract
Natural Language Processing (NLP) is today a very active field of research and innovation. Many applications need however big sets of data for supervised learning, suitably labelled for the training purpose. This includes applications for the Arabic language and its national dialects. However, such open access labeled data sets in Arabic and its dialects are lacking in the Data Science ecosystem and this lack can be a burden to innovation and research in this field. In this work, we present an open data set of social data content in several Arabic dialects. This data was collected from the Twitter social network and consists on +50K twits in five (5) national dialects. Furthermore, this data was labeled for several applications, namely dialect detection, topic detection and sentiment analysis. We publish this data as an open access data to encourage innovation and encourage other works in the field of NLP for Arabic dialects and social media. A selection of models were built using this data set and are presented in this paper along with their performances. ### Keywords: NLP · Open data · Supervised learning · Arab dialects.

#Data
The data was gathered by randomly scrapping tweets, from active users located in a predefined set of Arab countries, namely : Algeria, Egypt, Lebanon, Tunisia and Morocco. No limits were set for the date of the tweets nor for the exact location in the country. We used Selenium Python library to automate the web navigation and BeautifulSoup to scrap the tweets. The total number of tweets in the data set is 49,306. The tweets distribution per country is given in table 1.

![image](https://user-images.githubusercontent.com/59541945/134767292-00c8a7be-9926-4c32-962d-8bcb3dacbf59.png)

#Baseline models
Using our labeled data, we evaluate the performance of Arabic dialect identification, Arabic sentiment classification, and Arabic topic categorization systems
with the following machine learning models : Logistic Regression, SGD Classifier, Linear SVC and Naive bayes. We used grid search and pipelines to find the
best hyper-parametres.
#### Performance of the different tested models for dialect detection
![f](https://user-images.githubusercontent.com/59541945/134767519-c0affa95-8014-4626-9d7b-c42af8c7ebea.PNG)

#### Performance of the different tested models for topic detection
![f2](https://user-images.githubusercontent.com/59541945/134767521-84c5d4d4-ba8b-4d2a-9dcb-8dd9249c47e9.PNG)

#### Performance of the different tested models for Sentiment Analysis
![f3](https://user-images.githubusercontent.com/59541945/134767532-2140511b-7e16-457e-9cae-e7ae82311f5b.PNG)

# Conclusion
Conclusion In this work we presented a Labeled data set of 50K tweets in five (5) Arabic dialects. We presented the process of labeling this data for dialect detection, topic detection and sentiment analysis. We put this labeled data openly available for the research, startup and industrial community to build models for applications related to NLP for dialectal Arabic. We believe that initiatives such as ours can catalyse the innovation and the technological development of AI solutions in Arab countries like Morocco by removing the burden linked to the non availability of labeled data and to time-consuming tasks of collecting and manually labeling data. We also presented a set of Machine learning models that can be used as baseline models and to which future users of this data set can compare and aim to outperform by innovating in term of computational methods and algorithms. The labeled data set can be downloaded at [1], and all the implemented algorithms are available at

#Running the project

Ensure that you are in the project home directory. Create the machine learning model by running below command -
python model.py
This would create a serialized version of our model into a file model.pkl

Run app.py using below command to start Flask API
python app.py
By default, flask will run on port 5000.

Navigate to URL http://localhost:5000
You should be able to view the homepage as below : alt text

Enter valid numerical values in all 3 input boxes and hit Predict.

If everything goes well, you should be able to see the predcited salary vaule on the HTML page! alt text

You can also send direct POST requests to FLask API using Python's inbuilt request module Run the beow command to send the request with some pre-popuated values -
python request.py



