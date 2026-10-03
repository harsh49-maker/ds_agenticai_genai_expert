from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def train_optimized_classifier(X_train_scaled, y_train):
    """Runs a quick optimized classification setup to prevent background freezing loops."""
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Targeted grid configuration mapping settings perfectly
    param_grid = {
        "n_estimators":,
        "max_depth": [None, 6],
        "criterion": ['gini', 'entropy']
    }
    
    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        verbose=0,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    return grid_search.best_estimator_

def evaluate_classifier(model, X_test, y_test, model_name: str):
    """Prints the classification report layout table."""
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy : {accuracy:.2f}")
    print("Classification Report :")
    print(report)