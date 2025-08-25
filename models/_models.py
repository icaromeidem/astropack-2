from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def create_model(model_type, hp_combination = None):
    if model_type == "RF-REG":
        if hp_combination:
            n_features, n_trees, min_samples_leaf, bootstrap, max_features = hp_combination
    
            # Initialize the feature selector using the given hyperparameters
            FeatureSelector = RFE(
                estimator=DecisionTreeRegressor(),
                n_features_to_select=n_features,
                verbose=0,
                step=100,
            )
    
            # Initialize the random forest using the given hyperparameters
            RF = RandomForestRegressor(
                n_estimators=n_trees,
                bootstrap=bootstrap,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            )
        else:
            FeatureSelector = RFE(
                estimator=DecisionTreeRegressor()
            )
    
            RF = RandomForestRegressor()
            

        # Create a pipeline with the feature selector and the random forest
        pipeline = Pipeline(steps=[("Feature Selector", FeatureSelector), ("Model", RF)])
    
    elif model_type == "RF-CLA":
        if hp_combination:
            n_features, n_trees, min_samples_leaf, bootstrap, max_features, class_weight = hp_combination
    
            # Initialize the feature selector using the given hyperparameters
            FeatureSelector = RFE(
                estimator=DecisionTreeRegressor(),
                n_features_to_select=n_features,
                verbose=0,
                step=100,
            )
    
            # Initialize the random forest using the given hyperparameters
            RF = RandomForestClassifier(
                n_estimators=n_trees,
                bootstrap=bootstrap,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight = class_weight
            )
        else:
            FeatureSelector = RFE(
                estimator=DecisionTreeRegressor()
            )
            
            RF = RandomForestClassifier()

        # Create a pipeline with the feature selector and the random forest
        pipeline = Pipeline(steps=[("Feature Selector", FeatureSelector), ("Model", RF)])
    else:
        raise (ValueError("Modelo não suportado"))

    return pipeline
