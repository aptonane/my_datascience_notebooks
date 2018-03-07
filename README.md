
# My_datascience_notebooks <br>
### Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes. Presented in the form Jupyter Notebook.
# Contents:
# 1) House Sales in King County, USA  - Regularization and Selecting Models
Modelling housing prices using the king county home sales. This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015. <br>
Modelling housing prices using the king county home sales.  <br>
This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.<br>
In this notebook, we’ll explore ridge and lasso regression models. The idea is that by shrinking or regularizing the coefficients, prediction accuracy can be improved, variance can be decreased, and model interpretabily can also be improved.<br>
Regularization is a technique used in an attempt to solve the overfitting problem in statistical models. When someone wants to model a problem, he will might want to add more explaining variables. Thus model becomes more interesting and more complex and you find out that your result are quite good but not as perfect as you wish.<br>
So you continue adding more variables and your model will do good but it is probably overfitting, i.e. it will probably have poor prediction and generalization power: it sticks too much to the data and the model has probably learned the background noise while being fit. This isn't of course acceptable.
So how do you solve this? It is here where the regularization technique comes in handy.

# 2) Lending Club Unbalanced Dataset - Resampling Techniques
I am using in this notebook the publicly available dataset of Lending Club (2007 to 2015 file from Kaggle). It's a real world data set with a nice mix of categorical and continuous variables.(https://www.kaggle.com/wendykan/lending-club-loan-data/downloads/loan.csv).<br>
In this dataset there is a classification problem that is data imbalance between classes (Good Loans: 97.35% and Bad Loans: 2.65%) in target variable. In the other words, the class 1 (bad loans) has significantly lower representation in the trainning data relative to class 0 (good loans). The performance in minority class is that I care about in this work..<br>
Objectives:<br>
It is desired to have a high recall on the minority class (1) while maintaining a high precision on the majority class (0): Precison on 0 -> is the ratio tn / (tn + fn) where tn is the number of true negatives and fn the number of false negatives. It is intuitively the ability of the classifier not to label as bad loan (1) a value that is good loan (0). <br>
Recall on 1 -> is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. It is intuitively the ability of the classifier to find all the positive values (bad loans "1").<br>
In this notebook, I will show a number of resampling techniques to handle unbalanced datasets and see their effect on classification performance.<br>
For this problem I chose a classifier Random Forest that showed a good performance on the minority class. We will compare various techniques with respect to their effect on the recall on the minority class (1) and the precision on the majority class "0".<br>
credit: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=video&cd=1&cad=rja&uact=8&ved=0ahUKEwiEx_icppXZAhWHrFQKHQ4fDT0QtwIIKjAA&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D-Z1PaqYKC1w&usg=AOvVaw30WRonWtQ9z2XAqeAYEisX <br>

# 3) Medical Appointment No Shows: Why do 30% of patients miss their scheduled appointments?
A person makes a doctor appointment, receives all the instructions, and subsequently becomes a no-show. Since these no-shows bring huge losses to the Brazilian public finances, can I identify which variables are most important to predict the target variable (show or no-show)? Can I identify the most important variables in order to show the problems that should be fixed? Can I predict which person is most likely to no-show or show given the features related to that person?<br>
#### The Context
The number of no-shows in medical appointments in the health units of the city of Vitória-ES-Brazil reached 30.14% of the total number of appointments made in 2 years (2014/2015);<br>
This pattern is very similar to the brazilian national statistics;<br>
In Vitória alone, this rate represented an approximate loss of 19.5 million reais per year (5.9 million dollars); Considering all the operational costs of the scheduling, including sending SMS, confirmation links and the professionals involved, this cost reaches almost 2 million reais a month (about 606.000 dollars): a considerable waste!<br>
#### Inspiration
Why this data called my attention?<br>
It was collected in the city where I was born in Brazil (Vitória in Espírito Santo).<br>
It's interesting to work with data that you're familiar with. <br>
This data set was chosen from Kaggle's Dataset platform: https://www.kaggle.com/joniarroba/noshowappointments/data

# 4) Movie Revenue Prediction: What's the best model: Linear, Polynomial, Ridge, Lasso regression:<br>
#### Goal: Select best features and get the smallest rMSE and greatest r2
My intentions behind this notebook were to use the movie_metadata dataset to fit a regression model to predict gross. I used Linear, Polynomial, Ridge, Lasso regression and compared them based on r2 and rMSE evaluation metric.<br>
Data: https://www.kaggle.com/karrrimba/movie-metadatacsv  <br>

# 5) Admitted or not?:
The dataset contains four variables: admit, gre, gpa, and prestige:<br>
admit-> is a binary variable. It indicates whether or not a candidate was admitted into UCLA (admit = 1) our not (admit = 0). <br>
gre -> is the GRE score. GRE stands for Graduate Record Examination. <br>
gpa -> stands for Grade Point Average. <br>
prestige -> is the prestige of an applicant alta mater, with 1 being the highest (high prestige) and 4 as the lowest (not prestigious). <br>
#### Goal:<br>
The target variable admit indicates whether or not a candidate will be admitted into UCLA (admit = 1) our not (admit = 0). Minimizes False Positives means that the model will minimize incorrect flags as admit. Minimizes False Negatives means that the model will minimize incorrect flags as not admit. In this case, I will try to minimize the false positives because the cost of admit the wrong candidate could be higher than not admit the correct candidate. <br>

# 6) EDA (Exploratory Data Analysis) of Accidents in the Brazilian Federal Highways - (In Portuguese)
The Federal Highway Police in Brazil - PRF - serves about 70,000 km of federal highways and is distributed throughout the brazilian territory. <br>
Between 2007 and 2016 the accidents records was carried out through the system where the police responsible for the occurrence inserted the data concerning those involved, the place, the vehicles and the accident. In January 2017 this system (BR-Brazil) was discontinued and the PRF started to use a new system to record the occurrences of traffic accidents. This new system provides a succinct description of the variables present in datasets recorded from January 2017, mainly on the people involved in the accidents.<br>
This EDA is based on these datasets made available by theFederal Highway Police in Brazil.<br>
In the years of 2015, 2016 and 2017, about 14,000 people died on brazilian federal roads and 200,000 were injured. In total (dead, wounded and unharmed) about 510,000 were traffic victims. The numbers raised in this EDA are alarming and worrying.<br>
We also noted that the main cause of accidents was lack of attention. However, from the reports analyzed in parallel, the greatest cause of accidents has been the precarious conditions of conservation of the Brazilian roads. Perhaps the driver's mistakes regarding lack of attention (which should be redoubled when the road situation is inadequate), incompatible speed, alcohol intake, safety distance and disobedience to road signs are aggravated by the precarious condition of the roads, either the great cause of the excessive amount of accidents and deaths on the Brazilian roads.

# 7) NPL: Classifying Most Negative and Positives Reviews
Natural Language Processing (NLP) serves numerous use cases when dealing with text or unstructured text data. Let’s build a simple text classifier using Python’s Pandas, NLTK and Scikit-learn libraries.<br>
I'am going to train a machine learning algorithm to classify yelp reviews as either Most Negative (0) or Most Positive (1) reviews. 

# 8) NPL: Top ten "spammiest" words
Natural Language Processing (NLP) serves numerous use cases when dealing with text or unstructured text data. In this notebook, my goal is to find the top ten "spammiest" words in the dataset in yelp dataset.
