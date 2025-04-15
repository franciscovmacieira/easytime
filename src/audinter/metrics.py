import copy
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import torch.nn as nn
from difflib import get_close_matches


def algorithm_class_score(model, show_model_type=False):
    """
    Calculate a score for a given classifier based on a predefined scoring system.
    
    Parameters:
    model : object
        The classifier object for which the score is to be calculated.
    show_model_type : bool, optional
        If True, prints the name of the classifier. Default is False.

    Returns:
    (float, description)
        A normalized score (between 0 and 1) representing the quality or 
        expected performance of the classifier, or None if the classifier 
        type is not recognized.
    """

    # Dictionary mapping classifier names to their respective scores
    # https://github.com/ningxie1991/FederatedTrust/blob/develop/federatedTrust/configs/eval_metrics_v1.json
    alg_score = {
        "RandomForestClassifier": 4,
        "KNeighborsClassifier": 3,
        "SVC": 2,
        "GaussianProcessClassifier": 3,
        "DecisionTreeClassifier": 5,
        "MLPClassifier": 1,
        "AdaBoostClassifier": 3,
        "GaussianNB": 3.5,
        "QuadraticDiscriminantAnalysis": 3,
        "LogisticRegression": 4,
        "LinearRegression": 3.5,
    }

    # Get the name of the classifier class
    model_name = type(model).__name__
    description = "Level of explainability based on on literature research and qualitative analysis of each learning technique."

    if show_model_type: 
        print(model_name)

    # Check if the classifier name is in the alg_score dictionary
    if model_name in alg_score:
        exp_score = alg_score[model_name]
        return exp_score, description

    # Check if the classifier is a type of neural network
    if isinstance(model, tf.keras.Model) or isinstance(model, tf.Module) or isinstance(model, nn.Module):
        return 1, description  # Return a normalized score of 0.2 for neural networks
    
    # If classifier name is not found, try to find a close match
    close_matches = get_close_matches(model_name, alg_score.keys(), n=1, cutoff=0.6)
    if close_matches:
        exp_score = alg_score[close_matches[0]]
        return exp_score, description 
    
    # If no match is found, return None
    description =  f"No matching score found for '{model_name}'!"
    return None, description


def correlated_features_score(dataset, target_column=None, show_high_corr_feat=False):
    """
    Calculate a score based on the proportion of highly correlated features in a dataset.
    
    Parameters:
    dataset : pandas.DataFrame or array-like
        The input dataset containing features and possibly a target column.
    target_column : str, optional
        The name of the target column to be excluded from the correlation analysis. 
        If None, the last column of the dataset is assumed to be the target.
    show_high_corr_feat : bool, optional
        If True, prints the names of the removed features due to high correlation. Default is False.

    Returns:
    (float, description)
        A score representing the proportion of features that are not highly correlated. 
        The score ranges from 0 to 1, where 1 means no features were removed due to high correlation.
    """
    # Ensure the dataset is a pandas DataFrame
    if type(dataset) != 'pandas.core.frame.DataFrame':
        dataset = pd.DataFrame(dataset)

    # Make a deep copy of the dataset to avoid modifying the original data
    dataset = copy.deepcopy(dataset)
    
    # Exclude the target column from the features to be analyzed
    if target_column:
        X_test = dataset.drop(target_column, axis=1)
    else:
        X_test = dataset.iloc[:, :-1]

    # Retain only numeric data for correlation analysis
    X_test = X_test._get_numeric_data()
    # Compute the absolute correlation matrix
    corr_matrix = X_test.corr().abs()

    # Select the upper triangle of the correlation matrix to avoid duplicate pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    
    # Compute the average and standard deviation of the correlations in the upper triangle
    avg_corr = upper.values[np.triu_indices_from(upper.values, 1)].mean()
    std_corr = upper.values[np.triu_indices_from(upper.values, 1)].std()

    # Identify features with correlations greater than avg_corr + std_corr
    to_drop = [column for column in upper.columns if any(upper[column] > (avg_corr + std_corr))]
    if show_high_corr_feat: 
        print(f"Features with high correlation: {to_drop}")
    
    # Calculate the proportion of features removed due to high correlation
    pct_drop = len(to_drop) / len(X_test.columns)
    description = "Proportion of non-highly correlated features, correlation lesser than avg_corr + std_corr."
    return (round(1 - pct_drop, 2)), description  


def model_size(model, test_dataset=None):
    """
    Calculate the size of a machine learning model based on its type.
    
    Parameters:
    model : object
        The machine learning model whose size needs to be determined.
    test_dataset : array-like, optional
        A test dataset to be used for certain types of models.

    Returns:
    (int, description)
        The size of the model, which can be the number of parameters, 
        the number of nodes, the number of support vectors, or the 
        number of features seen in fit, depending on the model type.
    """
    
    # If the model is a TensorFlow Keras Model, return the count of parameters
    if isinstance(model, tf.keras.Model):
        description = "Returned number of parameters."
        return model.count_params(), description
    
    # If the model is a TensorFlow Module or PyTorch Module, return the sum of parameters
    elif isinstance(model, tf.Module) or isinstance(model, nn.Module):
        description = "Returned number of parameters."
        return sum(p.numel() for p in model.parameters()), description
    
    # If the model has 'estimators_', typically an ensemble model like RandomForest
    elif hasattr(model, 'estimators_'):
        count = 0
        for i, est in enumerate(model.estimators_):
            # If the estimator has a tree structure, add the number of nodes
            if hasattr(est, 'tree_'):
                description = "Returned number of total nodes."
                count += est.tree_.node_count

            # If the estimator has support vectors, add their count
            elif hasattr(est, 'n_support_'):
                description = "Returned number of total support vectors."
                count += sum(est.n_support_)
        return count, description
    
    # If the model is a Support Vector Classifier
    elif hasattr(model, 'SVC'):
        description = "Returned number of support vectors."
        return sum(model.n_support_), description

    # If the model has a tree structure, return the number of nodes
    elif hasattr(model, 'tree_'):
        description = "Returned number of nodes."
        return model.tree_.node_count, description
    
    # Return the number of features seen during fit if applicable
    elif hasattr(model, 'n_features_in_'):
        description = "Returned number of features seen during fit."
        return model.n_features_in_, description
    
    # If a test dataset is provided, return the number of features in the dataset
    elif test_dataset is not None:
        description = "Returned number of features in test_dataset."
        return test_dataset.shape[1], description
    
    # If none of the above conditions are met, return None
    else:
        description = "Unable to determine model size!"
        return None, description


def feature_importance_score(model):
    """
    Calculate the feature importance score for a given classifier.
    
    Parameters:
    model : object
        The classifier whose feature importance needs to be calculated.

    Returns:
    (float, description)
        The percentage of features that concentrate a specified threshold 
        of the total importance, or None if the classifier type is not supported.
    """
    distri_threshold = 0.5  # Threshold for cumulative distribution of feature importance

    # Lists of model names for regression and classification
    regression = ['LogisticRegression', 'LogisticRegression']  # Likely a typo, should be ['LogisticRegression']
    classifier = ['RandomForestClassifier', 'DecisionTreeClassifier']

    # Check if the classifier is a regression model
    if (type(model).__name__ in regression) or (get_close_matches(type(model).__name__, regression, n=1, cutoff=0.6)):
        # Get the feature importance for regression models (coefficients)
        importance = model.coef_.flatten()

        # Normalize the importance values to sum to 1
        total = 0
        for i in range(len(importance)):
            total += abs(importance[i])

        for i in range(len(importance)):
            importance[i] = abs(importance[i]) / total

    # Check if the classifier is a classification model
    elif (type(model).__name__ in classifier) or (get_close_matches(type(model).__name__, classifier, n=1, cutoff=0.6)):
        # Get the feature importance for classification models
        importance = model.feature_importances_
    
    else:
        description = "Classifier type is not applicable!"
        return None, description  

    # Sort the importance values in descending order
    indices = np.argsort(importance)[::-1]
    importance = importance[indices]
    
    # Calculate the percentage of features that concentrate distri_threshold percent of the total importance
    pct_dist = sum(np.cumsum(importance) < distri_threshold) / len(importance)
    
    description = f"Percentage of features that concentrate distri_threshold ({distri_threshold}) percent of the total importance."
    return pct_dist, description


def predict(model):
    """
    Get the prediction function of a model.
    
    Parameters:
    model : object
        The machine learning model.

    Returns:
    function
        The model's probability prediction function if available, 
        otherwise the standard prediction function.
    """
    return model.predict_proba if hasattr(model, 'predict_proba') else model.predict


def cv_shap_score(model, test_dataset):
    """
    Calculate the coefficient of variation (CV) of SHAP values for a classifier.
    
    Parameters:
    model : object
        The classifier for which SHAP values are to be calculated.
    test_dataset : pandas.DataFrame
        The test_dataset for which SHAP values are to be calculated.

    Returns:
    (float, description)
        The coefficient of variation of the absolute SHAP values, 
        or None if SHAP values cannot be calculated.
    """
    if type(test_dataset) != 'pandas.core.frame.DataFrame':
        test_dataset = pd.DataFrame(test_dataset)

    # Create a background dataset by sampling 100 samples from the provided test_dataset
    background = shap.sample(test_dataset, 100)

    # Initialize KernelExplainer with the prediction function and background dataset
    explainer = shap.KernelExplainer(predict(model), background)

    # Calculate SHAP values for the first instance in the dataset
    shap_values = explainer.shap_values(test_dataset.iloc[0])
  
    # Calculate the coefficient of variation of absolute SHAP values if they exist
    if shap_values is not None and len(shap_values) > 0:
        # Calculate the sum of absolute SHAP values for each class
        sums = np.array([abs(shap_values[i]).sum() for i in range(len(shap_values))])
        # Calculate the coefficient of variation (CV)
        cv = np.std(sums) / np.mean(sums)
        description = "Coefficient of variation of absolute SHAP values."
        return round(cv, 3), description
    
    else:
        description = "SHAP values cannot be calculated!"
        return None, description

def all_metrics(model, dataset, test_dataset=None):
    """
    This function calculates all the explainability and complexity model auditing metrics available.

    Parameters:
        model (object): The machine learning model for which the metrics is to be calculated.
        dataset (pandas.DataFrame): The input dataset containing features
        test_dataset (pandas.DataFrame): The test data for the metrics which this parameter is required.

    Returns:
        tuple: All audinter metrics available.
    """
    return algorithm_class_score(model),\
    correlated_features_score(dataset),\
    model_size(model),\
    feature_importance_score(model),\
    cv_shap_score(model, test_dataset)


