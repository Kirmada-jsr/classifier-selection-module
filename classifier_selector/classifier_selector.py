from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from mpa import mpa
from sklearn.model_selection import GridSearchCV

class Classifier_selector():
  def create_visualization(self, grid_search, conf_matrix, y_test, y_pred):
    """Create visualization plots for model performance"""
    fig = plt.figure(figsize=(20, 10))

    # Plot 1: Confusion Matrix
    ax1 = plt.subplot(121)
    sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    # Plot 2: Parameter Performance
    ax2 = plt.subplot(122)
    cv_results = pd.DataFrame(grid_search.cv_results_)
    top_scores_idx = cv_results['rank_test_score'].argsort()[:5]
    top_scores = cv_results.iloc[top_scores_idx]

    # Create parameter performance plot
    param_scores = pd.DataFrame({
        'Score': top_scores['mean_test_score'],
        'Parameters': [str(params) for params in
                      top_scores['params'].apply(lambda x: {k: v for k, v in x.items()})]
    })
    param_scores.plot(kind='bar', x='Parameters', y='Score', ax=ax2)
    ax2.set_title('Top 5 Parameter Combinations')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    return fig

  def get_feature_importances(self, model, feature_names):
    """Extract feature importances if the model supports them"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        return dict(zip(feature_names, np.abs(model.coef_[0])))
    return None

  def optimize_classifier(self, X_train, X_test, y_train, y_test, classifier_name='knn', custom_param_grid=None):
    """
    Perform grid search to find optimal parameters for various classifiers and evaluate the model.

    Parameters:
    X_train, X_test: Training and test features
    y_train, y_test: Training and test labels
    classifier_name: str, options: 'knn', 'perceptron', 'rf', 'dt', 'svm'
    custom_param_grid: dict, optional custom parameter grid

    Returns:
    dict: Best model, parameters, and performance metrics
    """
    # Define default parameter grids for each classifier
    default_param_grids = {
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'perceptron': {
            'penalty': [None, 'l1', 'l2', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'max_iter': [1000],
            'eta0': [0.1, 0.5, 1.0]
        },
        'rf': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'dt': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        'svm': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        },
        'mpa': {
            'learning_rate' :  [1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00] ,
            'epochs': [1, 5, 10, 50, 75, 100, 150, 350],
            'verbose': [False],
            'random_state':[46]
        }
    }

    # Dictionary mapping classifier names to their classes
    classifiers = {
        'knn': KNeighborsClassifier(),
        'perceptron': Perceptron(),
        'rf': RandomForestClassifier(),
        'dt': DecisionTreeClassifier(),
        'svm': SVC(),
        'mpa': mpa()
    }

    # Validate classifier name
    if classifier_name not in classifiers:
        raise ValueError(f"Classifier '{classifier_name}' not supported. Choose from: {list(classifiers.keys())}")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get classifier and parameter grid
    classifier = classifiers[classifier_name]
    param_grid = custom_param_grid if custom_param_grid else default_param_grids[classifier_name]

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1',
        verbose=1
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)

    # Make predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Create visualizations
    fig = self.create_visualization(grid_search, conf_matrix, y_test, y_pred)

    # Return results
    results = {
        'best_model': best_model,
        'best_parameters': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'visualization': fig,
        'feature_importances': self.get_feature_importances(best_model, X_train.columns)
    }

    return results