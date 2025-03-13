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
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc  # Add to imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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

    # Plot 2: ROC-AUC Curve
    ax2 = plt.subplot(122)
    best_model = grid_search.best_estimator_
    if hasattr(best_model, 'predict_proba'):  # Check if model gives probs
        y_prob = best_model.predict_proba(self.X_test_scaled)[:, 1]  # Positive class prob
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC-AUC Curve')
        ax2.legend(loc="lower right")
    else:
        ax2.text(0.5, 0.5, 'No ROC-AUC (No Probabilities)', ha='center', va='center')

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
    classifier_name: str, options: 'knn', 'perceptron', 'rf', 'dt', 'svm', 'ab', 'cb', 'ert', 'gb', 'lgbm', 'xgb'
    custom_param_grid: dict, optional custom parameter grid

    Returns:
    dict: Best model, parameters, and performance metrics
    """
    # Import necessary classifiers
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    # Define default parameter grids for each classifier
    default_param_grids = {
        'knn': {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        },
        'perceptron': {
            'penalty': [None, 'l1', 'l2'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'max_iter': [500, 1000],
            'tol': [1e-3, 1e-4],
            'early_stopping': [True],
            'validation_fraction': [0.1]
            },
        'rf': {
             'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8],
            'bootstrap': [True],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', None]
        },
        'dt': {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', None],
            'min_impurity_decrease': [0.0, 0.01, 0.05]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf'],
            'gamma': ['scale', 0.001, 0.01, 0.1],  # No 'auto'
            'degree': [2, 3],  # Only for poly
            'coef0': [0.0, 0.5],  # Only for poly/sigmoid
            'class_weight': ['balanced', None]
        },
        'mpa': {
            'learning_rate': [1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00],
            'epochs': [1, 5, 10, 50, 75, 100, 150, 350],
            'verbose': [False],
            'random_state': [46]
        },
        'ab': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'algorithm': ['SAMME']
        },
        'cb': {
            'iterations': [100, 200, 500],
            'learning_rate': [0.03, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 5, 10],
            'bagging_temperature': [0, 1],
            'boosting_type': ['Ordered', 'Plain']
        },
        'ert': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 0.5, 0.7],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', None]
        },
        'gb': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.7, 0.9],
            'criterion': ['friedman_mse']
        },
        'lgbm': {
            'n_estimators': [100, 200, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 100, 200, 300],
            'max_depth': [-1, 5, 10, 15, 20, 25],  # -1 means no limit
            'min_child_samples': [5, 10, 20, 50, 100],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
            'min_split_gain': [0, 0.1, 0.2, 0.5],
            'boosting_type': ['gbdt', 'dart', 'goss'],
            'objective': ['binary', 'multiclass'],
            'metric': ['multi_logloss', 'multi_error', None],
            'class_weight': ['balanced', None]
        },
        'xgb': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'gamma': [0, 0.1, 0.3],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9],
            'reg_alpha': [0, 0.1, 1.0],
            'booster': ['gbtree', 'dart']
        }
    }
    # Dictionary mapping classifier names to their classes
    classifiers = {
        'knn': KNeighborsClassifier(),
        'perceptron': Perceptron(),
        'rf': RandomForestClassifier(),
        'dt': DecisionTreeClassifier(),
        'svm': SVC(probability=True),
        'mpa': mpa(),
        'ab': AdaBoostClassifier(),
        'cb': CatBoostClassifier(verbose=0),
        'ert': ExtraTreesClassifier(),
        'gb': GradientBoostingClassifier(),
        'lgbm': LGBMClassifier(verbose=-1),
        'xgb': XGBClassifier(verbosity=0)
    }

    # Validate classifier name
    if classifier_name not in classifiers:
        raise ValueError(f"Classifier '{classifier_name}' not supported. Choose from: {list(classifiers.keys())}")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    self.X_test_scaled = scaler.transform(X_test)

    # Get classifier and parameter grid
    classifier = classifiers[classifier_name]
    param_grid = custom_param_grid if custom_param_grid else default_param_grids[classifier_name]

    # For resource-intensive parameter grids, consider implementing a more efficient approach
    if len(param_grid) > 5:
        print(f"Warning: Large parameter grid for {classifier_name}. Switching to RandomizedSearchCV")
        # Optional: Convert to RandomizedSearchCV instead for very large grids
        grid_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid,
                                       n_iter=100, cv=5, n_jobs=2, scoring='f1', verbose=1)
    else:
       grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=5,
        n_jobs=2,
        scoring='f1',
        verbose=1
    )
    print("n_jobs = 2")

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)

    # Make predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(self.X_test_scaled)

    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    mcc_score = matthews_corrcoef(y_test, y_pred)

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
        'mcc' :mcc_score,
        'feature_importances': self.get_feature_importances(best_model, X_train.columns)
    }

    return results