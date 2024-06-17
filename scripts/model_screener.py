"""
Functionality to quickly screen different models.
"""

from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np

class ModelScreener():
    """
    Perform model screening with standard hyperparameters.

    Parameters:
    - x_train: Training data
    - y_train: Training labels
    - models: A dictionary: {model names: model objects}
    - metrics: Scoring metrics
    - dataset_name: Optional, a string to identify the dataset
    - random_state: Optional, random seed for reproducibility
    - cv: Optional, number of cross-validation folds
    """
    
    def __init__(self, x_train, y_train, models, metrics, dataset_name="", 
                 random_state=42, cv=3):
        self.x_train = x_train
        self.y_train = y_train
        self.models = models
        self.metrics = metrics
        self.ds_name = dataset_name
        self.random_state = random_state
        self.cv = cv
        self.results = None
        self.results_df = None
    
    def screen_models(self) -> None:
        """
        Run the model screening process.
        """
        results = {}

        for model_name, model in self.models.items():
            # Perform screening
            model.fit(self.x_train, self.y_train)

            # Calculate cross-validation score
            score = cross_validate(
                model, 
                X=self.x_train, 
                y=self.y_train,
                scoring=self.metrics, 
                cv=self.cv, 
                n_jobs=-1
                )
            
            # Extract mean and standard deviation from score dictionary
            score_mean = np.mean(score["test_score"])
            score_stdev = np.std(score["test_score"])
            # Save all results of one particular model in results dict
            results[model_name] = (score_mean, score_stdev)
            
        self.results = results
    
    def transform_to_df(self) -> None:
        """
        Transform results dictionary in simple dataframe.
        """
        results_df = pd.DataFrame(self.results.values(), index=self.results.keys(), columns=['score_mean', 'score_stdev'])
        self.results_df = results_df