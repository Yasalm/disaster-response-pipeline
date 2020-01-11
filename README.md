# Data Scientist nano degree

## Disaster Response Pipeline Project


# Description 
Udacity's project titled as "Disaster Response Pipeline Project" aim to train a model to classifiy Disaster messagaes.It aims to use Natural Language Processing to categorise disatore tweets. The provide data contained pre-labeld tweets.




# Software and Libaries
- Python + 3.5
- NumPy
- pandas
- scikit-learn (v0.17)
- SQLalchemy
- NLTK
- Plotly
- flask
- argparse

# Screenshots 
1. Main Page. Shows Graph of training data. e.g diffrenet types of categories and genre. 
![alt text](https://github.com/yasir-almutairi/disaster-response/blob/master/screenshot/Screen%20Shot%202020-01-11%20at%207.35.57%20PM.png)

2. Here is an example of predicted categories of a distress message.

![alt text](https://github.com/yasir-almutairi/disaster-response/blob/master/screenshot/Screen%20Shot%202020-01-11%20at%207.45.14%20PM.png)

![alt text](https://github.com/yasir-almutairi/disaster-response/blob/master/screenshot/Screen%20Shot%202020-01-11%20at%207.45.23%20PM.png)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
