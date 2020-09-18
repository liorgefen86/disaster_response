# Disaster Response Pipeline Project

## Objective
The objective of the project is to create a machine learning-based tool that
 can, given a text input, select the appropriate categorises for it. The
tool should be able to assign several categorises to a text input (out of 36
 categories)
 
## Requirements

To install the necessary packages:

`pip install -r requirements.txt`


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse xx models/classifier_name`
        
        - `xx`: is a classifier 

2. Run the following command in the app's directory to run your web app.

    `python run.py`

3. In your favourite browser, enter below url:

`localhost:3001`
