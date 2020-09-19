# Disaster Response Pipeline Project

## Objective
The objective of the project is to create a machine learning-based tool that
 can, given a text input, select the appropriate categorises for it. The
tool should be able to assign several categorises to a text input (out of 36
 categories)
 
## Requirements

The project was built using python `3.8.5`.

To install the necessary packages:

`pip install -r requirements.txt`


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
        `python data_processing/process_data.py data/disaster_messages.csv data
        /disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python train_classifier.py data/DisasterResponse xx models/classifier_file`
        
        - `xx`: is a classifier string, such as:
            - `ad`: AdaBoost classifier
            - `rf`: RandomForest classifier
            - `dt`: DecisionTree classifier
        - `classifier_file`: name of the file to save the model in (without the
         extension)

2. Run the following command in the root's directory to run your web app.

    `python app/run.py`

3. In your favourite browser, enter below url:

`localhost:3001`
