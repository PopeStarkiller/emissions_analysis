![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/1_nPcdyVwgcuEZiEZiRqApug.jpeg?raw=true)

# Emissions Data Analysis with Spark, Tensorflow, and Prophet 

created by Robert Gramlich


![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f736f6369616c2e706e67.png?raw=true)

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/fb-prophet.png?raw=true)

<!-- ABOUT THE PROJECT -->
## About The Project

## Project Description

The purpose of this project was to showcase various skills producing predictive AI models.
It is important to note there was no intent on defending the fidelity of the results from the data. 
The datasets could not be verified and were from unofficial kaggle sources.
I chose to do a raw SQLAlchemy query conversion to showcase my understanding of how they work.
I chose to utilize spark dataframes to show my knowledge base with them and my ability to convert raw data to them.
I chose to use deep learning models to display my ability converting data types to work with the deep models.

## The Process
I bought a virtual private server from GoDaddy to begin the whole endeavor. I had used AWS RDS servers in the past but found
they can become increasingly expensive in an expedited fashion.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/andromeda.PNG?raw=true)

The next phase was to connect to the VPS. There are number of ways to achieve this objective. I chose PuTTy. 
PuTTy allows you to access in a remote fashion servers you have set up.
![Screenshot of putty](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/putty.PNG?raw=true)

After connecting to the VPS I had to create the database, user, access information, and edit traffic permissions
all through the ubuntu remote shell.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/linux.PNG?raw=true)

I loaded the kaggle datasets via SQLAlchemy on a seperate notebook. The reason it isn't included is the very reason I
decided to use a VPS in the first place.  The datasets are combined over 1.5 GB in size and thus will not be added to github.
Checking here I show how the data is present via PGAdmin.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/pgadmin.PNG?raw=true)

The Pyspark and Prophet libraries are very sensitive to windows.  Windows native compiling is somewhat flawed for these 
libraries. So, the best solution is to start your notebook in an anaconda shell. They way the anaconda envorinments compile,
especially as it relates to c++, is vastly superior.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/anaconda2.PNG?raw=true)

Pyspark requires declaration of environment variables to the code execution. To handle this I created a few lines
of code to detect what machine I was using so as to produce the correct target locations.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/env.PNG?raw=true)

The next phase was selecting the countries you desired to look at as far as trade groupings are concerned.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/inputs.PNG?raw=true)

The inputs you choose from the list of available countries then gets fed into the functions I created that generate
various spark dataframes.  These functions are table specific and create the dataframe from raw SQLAlchemy queries.  
The important part was to make sure all the constraints were met so that the data would be sensible for comparison i.e. column names, column types.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/sparks.PNG?raw=true)

During the merging process, all the dataframes were concatenated through the spark specific formats. To make the data
sensible for the next phases it was required for the columns to be logged. This was to reduce the variability and 
reduce the mean squared error that results from generating the models. The dataframes were also filtered by year min and max.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/merge.PNG?raw=true)

The next phase was to generate multiple deep learning models based upon the merged dataframe.  Each category will be
the y value once and an x value for the other instances.  What this does is creates specific normalizors and models for
use in comparison later.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/models.PNG?raw=true)

The next phase was to generate pandas dataframes from the spark dataframes. Prophet only takes the time series values
and the y value you're attempting to predict. This can only be passed as a pandas dataframe with specific column names.
Just like with the TensorFlow models, I created predictions for each column as though it were a y value once.
The predictions are made for each country's data. It makes no sense to do it for the merged dataframe as it contains
multiple values for the same year that are not sequential for the data. So, looking at each country and each column dataframes
and models get created for every prediction. These are broke up into the yhat, yhat_lower, yhat_upper categories.
These categories represent the boundry regions and their mean for the model's predictions. The number of predictions is 
determined by your input and the number represent years in the future to be predicted.                                   
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/prophets.PNG?raw=true)

After creating the models and their predictions, the dataframes are treated as values to be plugged in to the tensorflow
models. Holding each column as the y value at least once, just as before, the x values that represent the prophet predictions
are then plugged into the correct corresponding deep learning model to generate its predictions.  This process is done for
each country, for each corresponding y value, and for each yhat category. This allows for absolute comparison across all
boundries and data types.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/predictions.PNG?raw=true)

All of the information has up to this point been stored inside of a dictionary. This dictionary is then called on a final time
to generate to comparison you'd like to see.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/compare.PNG?raw=true)

Inputs are given to determine the country, the yhat, and the y value for the final comparison. The corresponding
r value and slope are also produced by the function. The reason this method was chosen is because, in my opinion,
the strength of the relationship and the slope between the deep model predictions and the prophet predictions tell a complete
story.  The absolute ideal slope is 1. This implies a 1 to 1 ration which is what you desire. Slope alone doesn't help you
though. The data points may be disparate and unrelated but the linear fit may produce you a slope of 1.  So it is important
that the relationship be strong. Another methodology might be mean absolute difference, but I find it an arbitrary choice.
Conversely, a strong relationship is meaningless if the slope is not close to 1. Both values need to be close to one otherwise
you can conclude with certainty that the models poorly predict the future. It is important to note that the deep learning model 
isn't meant by its nature to predict the future. The deep learning model is meant to simply predict a value based upon inputs.
Through trial and error I found the farther out the predictions were attempted to be made the worse the comparisons got.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/final1.PNG?raw=true)
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/final2.PNG?raw=true)

## Conclusion
Once again I'd like to state the purpose of this project was simply to showcase the my CRUD/ETL abilities, my understanding of data science,
my understanding of tensorflow, my understanding of prophet and time series data, and my ability to create complex functions.  Any conclusions
one might be able to draw from this data as far as predictions go, would be built on a foundation of uncertain data.

## Data

I chose various kaggle datasets that revolved around trade and emissions. 

### Built With

* PostgreSQL/SQLAlchemy
* Pandas
* Spark
* Numpy
* Tensorflow
* Prophet
* Ubuntu

<!-- CONTACT -->
## Contact
Robert Gramlich
robert.gramlich.ii@gmail.com

[Project Link](https://github.com/PopeStarkiller/emissions_analysis)
