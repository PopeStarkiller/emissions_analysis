# Emissions Data Analysis with Spark, Tensorflow, and Prophet 

created by Robert Gramlich

![Twitter Sentiment Image](https://miro.medium.com/max/1400/1*0P55fknrgWKxG0gfwAGCvw.png)

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
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

## Project Description

I bought a virtual private server from GoDaddy to begin the whole endeavor. I had used AWS RDS servers in the past but found
they can become increasingly expensive in an expedited fashion.
![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/andromeda.PNG?raw=true)

The next phase was to connect to the VPS. There are number of ways to achieve this objective. I chose PuTTy. 
PuTTy allows you to access in a remote fashion servers you have set up.

![Screenshot of putty](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/putty.PNG?raw=true)

After connecting to the VPS I had to create the database, user, access information, and edit traffic permissions
all through the ubuntu remote shell.

![Screenshot of vps](https://github.com/PopeStarkiller/emissions_analysis/blob/main/static/images/linux.PNG?raw=true)

## Data



### Built With

* PostgreSQL/SQLAlchemy
* Pandas - tensorflow
* Flask
* HTML
* JavaScript - D3

<!-- GETTING STARTED -->
## Getting Started


   ```



<!-- USAGE EXAMPLES -->
## Usage

User grabs a tweet by giving the keyword selection an input. Once the tweet has been loaded onto the webpage, the user selects whether they consider the tweet to be neutral, positive, or negative in sentiment. Based upon the sum of data collected, various models will make predictions.  The adjudication model will always make a prediction. The other models however will only make predictions if the tweet is declared to have sentiment. I feel it illogical to use resources predicting sentiment on a tweet that has none. 

![Screenshot of Sentiment Analysis Application](https://github.com/sunwoo-kim20/sentiment-analysis-final-project/blob/main/static/images/voting_page.png?raw=true)


![Screenshot of Neural Network Structure](https://github.com/sunwoo-kim20/sentiment-analysis-final-project/blob/main/static/images/the_model.PNG?raw=true)



<!-- CONTACT -->
## Contact


[Project Link](https://github.com/PopeStarkiller/NLP)
