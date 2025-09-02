
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_important_features(X, y, n_features_to_save=10, params=None, random_state=42, n_estimators=100):
    """
    Avalia a importância das features usando RandomForestRegressor.
    Retorna as n_features_to_save mais importantes.
    """
    if params is not None:
        rf = RandomForestRegressor(**params)
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_names = np.array(X.columns)
    # Ordenar por importância decrescente
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]
    # Selecionar as n_features_to_save mais importantes
    selected_features = sorted_features[:n_features_to_save]
    # DataFrame para visualização
    cumulative_importance = np.cumsum(sorted_importances)
    df_feat = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importances,
        'cumulative_importance': cumulative_importance
    })
    return selected_features.tolist(), df_feat


def get_important_features_xgb(X, y, n_features_to_save=10, params=None, random_state=42, n_estimators=100):
    """
    Avalia a importância das features usando XGBRegressor.
    Retorna as n_features_to_save mais importantes.
    """
    # Se params for fornecido, use as configurações do usuário, mas garanta defaults para random_state, n_estimators, n_jobs se não estiverem presentes
    if params is not None:
        user_params = params.copy()
        if 'random_state' not in user_params:
            user_params['random_state'] = random_state
        if 'n_estimators' not in user_params:
            user_params['n_estimators'] = n_estimators
        if 'n_jobs' not in user_params:
            user_params['n_jobs'] = -1
        if 'verbosity' not in user_params:
            user_params['verbosity'] = 0
        xgb = XGBRegressor(**user_params)
    else:
        xgb = XGBRegressor(n_estimators=n_estimators, random_state=random_state, verbosity=0, n_jobs=-1)
    xgb.fit(X, y)
    importances = xgb.feature_importances_
    feature_names = np.array(X.columns)
    # Ordenar por importância decrescente
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]
    # Selecionar as n_features_to_save mais importantes
    selected_features = sorted_features[:n_features_to_save]
    # DataFrame para visualização
    cumulative_importance = np.cumsum(sorted_importances)
    df_feat = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importances,
        'cumulative_importance': cumulative_importance
    })
    return selected_features.tolist(), df_feat
