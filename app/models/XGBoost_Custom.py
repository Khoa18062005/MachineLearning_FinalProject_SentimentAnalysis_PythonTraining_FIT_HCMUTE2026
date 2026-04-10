import numpy as np
import pandas as pd
import pickle
import time
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# =========================
# LOAD / SAVE ARTIFACTS
# =========================
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    return (
        model_data['base_score_logit'],
        model_data['trees'],
        model_data['learning_rate'],
        model_data['feature_names'],
        model_data['classes'],
        model_data['threshold'],
        model_data.get('training_time_sec', 0.0),
        model_data.get('params', {})
    )


def load_vectorizer(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# =========================
# HÀM TOÁN HỌC HỖ TRỢ
# =========================
def sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


# =========================
# PHẦN XỬ LÝ TEXT -> VECTOR
# =========================
def fit_text_vectorizer(X_train_text, max_features=2000, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    X_vec = vectorizer.fit_transform(pd.Series(X_train_text).fillna('').astype(str)).astype(np.float32).toarray()
    feature_names = list(vectorizer.get_feature_names_out())
    X_df = pd.DataFrame(X_vec, columns=feature_names)
    return vectorizer, X_df, feature_names


def transform_text_with_vectorizer(X_text, vectorizer):
    X_vec = vectorizer.transform(pd.Series(X_text).fillna('').astype(str)).astype(np.float32).toarray()
    feature_names = list(vectorizer.get_feature_names_out())
    X_df = pd.DataFrame(X_vec, columns=feature_names)
    return X_df


# =========================
# PHẦN XỬ LÝ FEATURE / LABEL
# =========================
def prepare_features(X, feature_names=None):
    if isinstance(X, pd.Series):
        X = X.to_frame().T

    if isinstance(X, pd.DataFrame):
        if feature_names is not None:
            missing_cols = [col for col in feature_names if col not in X.columns]
            if missing_cols:
                raise ValueError(f'Thiếu cột đầu vào: {missing_cols}')
            X = X[feature_names]
        X_array = X.to_numpy(dtype=np.float32)
        names = list(X.columns) if feature_names is None else list(feature_names)
        return X_array, names

    X_array = np.asarray(X, dtype=np.float32)
    if X_array.ndim == 1:
        X_array = X_array.reshape(1, -1)

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
    else:
        if X_array.shape[1] != len(feature_names):
            raise ValueError('Số lượng feature không khớp với feature_names đã lưu.')

    return X_array, list(feature_names)


def encode_target(y):
    if isinstance(y, pd.Series):
        y_array = y.to_numpy()
    else:
        y_array = np.asarray(y)

    classes = list(pd.unique(y_array))
    if len(classes) != 2:
        raise ValueError('XGB Custom bản này chỉ hỗ trợ binary classification (2 lớp).')

    label_to_int = {classes[0]: 0, classes[1]: 1}
    y_encoded = np.array([label_to_int[label] for label in y_array], dtype=np.float32)
    return y_encoded, classes


# =========================
# PHẦN TREE BOOSTING
# =========================
def compute_leaf_weight(G, H, lambda_reg):
    return -G / (H + lambda_reg)


def compute_split_gain(G_left, H_left, G_right, H_right, G_total, H_total, lambda_reg, gamma):
    left_score = (G_left ** 2) / (H_left + lambda_reg)
    right_score = (G_right ** 2) / (H_right + lambda_reg)
    parent_score = (G_total ** 2) / (H_total + lambda_reg)
    return 0.5 * (left_score + right_score - parent_score) - gamma


def find_best_split(X, g, h, feature_indices, params):
    best_split = None
    G_total = np.sum(g)
    H_total = np.sum(h)
    max_thresholds = params.get('max_thresholds_per_feature', 8)

    for feature_idx in feature_indices:
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        if len(unique_values) <= 1:
            continue

        if len(unique_values) > max_thresholds + 1:
            quantiles = np.linspace(0.05, 0.95, max_thresholds)
            thresholds = np.unique(np.quantile(feature_values, quantiles))
        else:
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            left_count = np.sum(left_mask)
            right_count = np.sum(right_mask)

            if left_count < params['min_samples_leaf'] or right_count < params['min_samples_leaf']:
                continue

            G_left = np.sum(g[left_mask])
            H_left = np.sum(h[left_mask])
            G_right = np.sum(g[right_mask])
            H_right = np.sum(h[right_mask])

            if H_left < params['min_child_weight'] or H_right < params['min_child_weight']:
                continue

            gain = compute_split_gain(
                G_left, H_left,
                G_right, H_right,
                G_total, H_total,
                params['lambda_reg'],
                params['gamma']
            )

            if gain <= 0:
                continue

            if best_split is None or gain > best_split['gain']:
                best_split = {
                    'feature_idx': int(feature_idx),
                    'threshold': float(threshold),
                    'gain': float(gain),
                    'left_mask': left_mask,
                    'right_mask': right_mask
                }

    return best_split


def build_tree(X, g, h, depth, params):
    G = np.sum(g)
    H = np.sum(h)
    leaf_weight = compute_leaf_weight(G, H, params['lambda_reg'])

    node = {
        'is_leaf': True,
        'weight': float(leaf_weight),
        'n_samples': int(len(g)),
        'depth': int(depth)
    }

    if depth >= params['max_depth']:
        return node

    if len(g) < params['min_samples_split']:
        return node

    best_split = find_best_split(X, g, h, params['feature_indices'], params)

    if best_split is None:
        return node

    left_mask = best_split['left_mask']
    right_mask = best_split['right_mask']

    left_node = build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1, params)
    right_node = build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1, params)

    return {
        'is_leaf': False,
        'feature_idx': best_split['feature_idx'],
        'feature_name': params['feature_names'][best_split['feature_idx']],
        'threshold': best_split['threshold'],
        'gain': best_split['gain'],
        'n_samples': int(len(g)),
        'depth': int(depth),
        'left': left_node,
        'right': right_node
    }


def predict_tree_single(x_row, node):
    if node['is_leaf']:
        return node['weight']

    if x_row[node['feature_idx']] <= node['threshold']:
        return predict_tree_single(x_row, node['left'])
    return predict_tree_single(x_row, node['right'])


def predict_tree_batch(X, tree):
    outputs = np.zeros(X.shape[0], dtype=np.float32)
    for i in range(X.shape[0]):
        outputs[i] = predict_tree_single(X[i], tree)
    return outputs


# =========================
# HỖ TRỢ TĂNG TỐC KHI TUNE
# =========================
def sample_series(X, y, sample_ratio=1.0, random_state=42):
    X_series = pd.Series(X).fillna('').astype(str).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)

    if sample_ratio >= 1.0:
        return X_series, y_series

    n_samples = len(X_series)
    sample_size = max(1, int(n_samples * sample_ratio))
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(n_samples, size=sample_size, replace=False))
    return X_series.iloc[indices].reset_index(drop=True), y_series.iloc[indices].reset_index(drop=True)


# =========================
# TRAIN / PREDICT CHO TEXT
# =========================
def train_XGB_Custom(
    X_train_text,
    y_train,
    max_features=2000,
    ngram_range=(1, 1),
    n_estimators=20,
    learning_rate=0.1,
    max_depth=3,
    lambda_reg=1.0,
    gamma=0.0,
    min_child_weight=1.0,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,
    colsample_bytree=0.8,
    threshold=0.5,
    random_state=42,
    max_thresholds_per_feature=8,
    save_artifacts=True,
    model_filename='XGB_model_custom.pkl',
    vectorizer_filename='XGB_vectorizer.pkl',
    return_vectorizer=False,
    verbose=True
):
    start_time = time.time()
    np.random.seed(random_state)

    vectorizer, X_train_df, feature_names = fit_text_vectorizer(
        X_train_text=X_train_text,
        max_features=max_features,
        ngram_range=ngram_range
    )

    X, feature_names = prepare_features(X_train_df)
    y, classes = encode_target(y_train)

    n_samples, n_features = X.shape

    pos_rate = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
    base_score_logit = float(np.log(pos_rate / (1 - pos_rate)))
    raw_scores = np.full(n_samples, base_score_logit, dtype=np.float32)

    trees = []

    for i in range(n_estimators):
        p = sigmoid(raw_scores)
        g = p - y
        h = p * (1.0 - p)

        if 0 < subsample < 1.0:
            sample_size = max(1, int(subsample * n_samples))
            row_idx = np.random.choice(n_samples, size=sample_size, replace=False)
        else:
            row_idx = np.arange(n_samples)

        if 0 < colsample_bytree < 1.0:
            feature_size = max(1, int(colsample_bytree * n_features))
            feature_indices = np.random.choice(n_features, size=feature_size, replace=False)
        else:
            feature_indices = np.arange(n_features)

        tree_params = {
            'max_depth': max_depth,
            'lambda_reg': lambda_reg,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'feature_indices': feature_indices,
            'feature_names': feature_names,
            'max_thresholds_per_feature': max_thresholds_per_feature
        }

        tree = build_tree(X[row_idx], g[row_idx], h[row_idx], depth=0, params=tree_params)
        tree_output = predict_tree_batch(X, tree)
        raw_scores += learning_rate * tree_output
        trees.append(tree)

        if verbose:
            elapsed = time.time() - start_time
            done = i + 1
            progress_percent = round(done / n_estimators * 100, 2)
            avg_time_per_tree = elapsed / done
            eta_sec = round(avg_time_per_tree * (n_estimators - done), 2)

            print(
                f'[{done}/{n_estimators}] {progress_percent}% | elapsed: {round(elapsed, 2)}s | ETA: {eta_sec}s',
                flush=True
            )

    training_duration = round(time.time() - start_time, 2)
    model_data = {
        'base_score_logit': base_score_logit,
        'trees': trees,
        'learning_rate': learning_rate,
        'feature_names': feature_names,
        'classes': classes,
        'threshold': threshold,
        'params': {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'lambda_reg': lambda_reg,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'max_thresholds_per_feature': max_thresholds_per_feature
        },
        'training_time_sec': training_duration
    }

    if save_artifacts:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_filename)
        vectorizer_path = os.path.join(current_dir, vectorizer_filename)

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

    if return_vectorizer:
        return model_data, vectorizer

    return model_data


def predict_XGB_Custom(
    text_input,
    base_score_logit,
    trees,
    learning_rate,
    feature_names,
    classes,
    threshold=0.5,
    vectorizer=None
):
    if vectorizer is None:
        raise ValueError('Cần truyền vectorizer khi dự đoán text cho XGB Custom.')

    if isinstance(text_input, str):
        X_input = [text_input]
    else:
        X_input = text_input

    X_text_df = transform_text_with_vectorizer(X_input, vectorizer)
    X, _ = prepare_features(X_text_df, feature_names=feature_names)
    raw_scores = np.full(X.shape[0], base_score_logit, dtype=np.float32)

    for tree in trees:
        raw_scores += learning_rate * predict_tree_batch(X, tree)

    prob_positive = sigmoid(raw_scores)
    prob_negative = 1.0 - prob_positive
    pred_int = (prob_positive >= threshold).astype(int)
    predictions = np.where(pred_int == 1, classes[1], classes[0])

    results_df = pd.DataFrame({
        classes[0]: prob_negative,
        classes[1]: prob_positive
    })

    if X.shape[0] == 1:
        return predictions[0], results_df.iloc[0].to_dict()

    return predictions, results_df


def evaluate_XGB_Custom(X_eval_text, y_eval, model_data, vectorizer):
    y_pred = []
    for text in X_eval_text:
        label, _ = predict_XGB_Custom(
            text_input=text,
            base_score_logit=model_data['base_score_logit'],
            trees=model_data['trees'],
            learning_rate=model_data['learning_rate'],
            feature_names=model_data['feature_names'],
            classes=model_data['classes'],
            threshold=model_data['threshold'],
            vectorizer=vectorizer
        )
        y_pred.append(label)

    acc = accuracy_score(y_eval, y_pred)
    return float(acc), y_pred


# =========================
# AUTO TUNE CHO XGB CUSTOM - BẢN NHANH HƠN
# =========================
def sample_random_params(search_space, rng):
    return {
        'max_features': rng.choice(search_space['max_features']),
        'ngram_range': rng.choice(search_space['ngram_range']),
        'n_estimators': rng.choice(search_space['n_estimators']),
        'learning_rate': rng.choice(search_space['learning_rate']),
        'max_depth': rng.choice(search_space['max_depth']),
        'lambda_reg': rng.choice(search_space['lambda_reg']),
        'gamma': rng.choice(search_space['gamma']),
        'min_child_weight': rng.choice(search_space['min_child_weight']),
        'min_samples_split': rng.choice(search_space['min_samples_split']),
        'min_samples_leaf': rng.choice(search_space['min_samples_leaf']),
        'subsample': rng.choice(search_space['subsample']),
        'colsample_bytree': rng.choice(search_space['colsample_bytree']),
        'threshold': rng.choice(search_space['threshold']),
        'max_thresholds_per_feature': rng.choice(search_space['max_thresholds_per_feature'])
    }


def train_XGB_Custom_AutoTune(
    X_train_text,
    y_train,
    X_val_text,
    y_val,
    n_trials=8,
    random_state=42,
    search_space=None,
    retrain_on_train_val=True,
    early_stop_patience=3,
    tune_train_sample_ratio=0.7,
    tune_val_sample_ratio=1.0,
    model_filename='XGB_model_custom.pkl',
    vectorizer_filename='XGB_vectorizer.pkl'
):
    total_start_time = time.time()
    rng = random.Random(random_state)

    if search_space is None:
        search_space = {
            'max_features': [1500, 2000, 3000],
            'ngram_range': [(1, 1), (1, 2)],
            'n_estimators': [15, 25, 35],
            'learning_rate': [0.05, 0.08, 0.1],
            'max_depth': [2, 3, 4],
            'lambda_reg': [0.5, 1.0, 2.0],
            'gamma': [0.0, 0.1],
            'min_child_weight': [1.0, 2.0],
            'min_samples_split': [2, 4],
            'min_samples_leaf': [1, 2],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'threshold': [0.5],
            'max_thresholds_per_feature': [6, 8]
        }

    X_train_text = pd.Series(X_train_text).fillna('').astype(str)
    y_train = pd.Series(y_train).reset_index(drop=True)
    X_val_text = pd.Series(X_val_text).fillna('').astype(str)
    y_val = pd.Series(y_val).reset_index(drop=True)

    X_tune_train, y_tune_train = sample_series(
        X_train_text, y_train,
        sample_ratio=tune_train_sample_ratio,
        random_state=random_state
    )
    X_tune_val, y_tune_val = sample_series(
        X_val_text, y_val,
        sample_ratio=tune_val_sample_ratio,
        random_state=random_state + 999
    )

    best_score = -1.0
    best_trial_model_data = None
    best_trial_vectorizer = None
    best_params = None
    tried_configs = set()
    no_improve_count = 0

    for trial in range(1, n_trials + 1):
        params = sample_random_params(search_space, rng)
        config_key = tuple(sorted((k, str(v)) for k, v in params.items()))

        retry_count = 0
        while config_key in tried_configs and retry_count < 20:
            params = sample_random_params(search_space, rng)
            config_key = tuple(sorted((k, str(v)) for k, v in params.items()))
            retry_count += 1

        tried_configs.add(config_key)
        params['random_state'] = random_state + trial

        trial_model_data, trial_vectorizer = train_XGB_Custom(
            X_train_text=X_tune_train,
            y_train=y_tune_train,
            max_features=params['max_features'],
            ngram_range=params['ngram_range'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            lambda_reg=params['lambda_reg'],
            gamma=params['gamma'],
            min_child_weight=params['min_child_weight'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            threshold=params['threshold'],
            random_state=params['random_state'],
            max_thresholds_per_feature=params['max_thresholds_per_feature'],
            save_artifacts=False,
            return_vectorizer=True,
            verbose=False
        )

        trial_score, _ = evaluate_XGB_Custom(
            X_eval_text=X_tune_val,
            y_eval=y_tune_val,
            model_data=trial_model_data,
            vectorizer=trial_vectorizer
        )

        print(
            f'[Trial {trial}/{n_trials}] val_accuracy = {round(trial_score * 100, 2)}% | params = {params}',
            flush=True
        )

        if trial_score > best_score:
            best_score = trial_score
            best_trial_model_data = trial_model_data
            best_trial_vectorizer = trial_vectorizer
            best_params = dict(params)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if early_stop_patience is not None and no_improve_count >= early_stop_patience:
            print(
                f'Dừng sớm sau {trial} trials vì {early_stop_patience} trial liên tiếp không cải thiện.',
                flush=True
            )
            break

    if best_trial_model_data is None:
        raise RuntimeError('Không tìm được bộ tham số phù hợp trong quá trình auto tune.')

    if retrain_on_train_val:
        X_final_text = pd.concat([
            pd.Series(X_train_text).reset_index(drop=True),
            pd.Series(X_val_text).reset_index(drop=True)
        ], ignore_index=True)
        y_final = pd.concat([
            pd.Series(y_train).reset_index(drop=True),
            pd.Series(y_val).reset_index(drop=True)
        ], ignore_index=True)

        final_model_data, final_vectorizer = train_XGB_Custom(
            X_train_text=X_final_text,
            y_train=y_final,
            max_features=best_params['max_features'],
            ngram_range=best_params['ngram_range'],
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            lambda_reg=best_params['lambda_reg'],
            gamma=best_params['gamma'],
            min_child_weight=best_params['min_child_weight'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            threshold=best_params['threshold'],
            random_state=best_params['random_state'],
            max_thresholds_per_feature=best_params['max_thresholds_per_feature'],
            save_artifacts=False,
            return_vectorizer=True,
            verbose=True
        )
    else:
        final_model_data = best_trial_model_data
        final_vectorizer = best_trial_vectorizer

    total_training_time = round(time.time() - total_start_time, 2)

    final_model_data['best_params'] = best_params
    final_model_data['best_score_val'] = round(best_score, 6)
    final_model_data['n_trials'] = n_trials
    final_model_data['retrain_on_train_val'] = retrain_on_train_val
    final_model_data['training_time_sec'] = total_training_time
    final_model_data['params'] = {
        **final_model_data.get('params', {}),
        'best_params': best_params,
        'best_score_val': round(best_score, 6),
        'n_trials': n_trials,
        'retrain_on_train_val': retrain_on_train_val,
        'early_stop_patience': early_stop_patience,
        'tune_train_sample_ratio': tune_train_sample_ratio,
        'tune_val_sample_ratio': tune_val_sample_ratio
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_filename)
    vectorizer_path = os.path.join(current_dir, vectorizer_filename)

    with open(model_path, 'wb') as f:
        pickle.dump(final_model_data, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(final_vectorizer, f)

    return final_model_data


if __name__ == '__main__':
    from app.services.ml_service import get_clean_datasets_for_training

    df_train, df_val, df_test = get_clean_datasets_for_training()

    X_train = df_train['text'].fillna('')
    y_train = df_train['target']

    X_val = df_val['text'].fillna('')
    y_val = df_val['target']

    model_metadata = train_XGB_Custom_AutoTune(
        X_train_text=X_train,
        y_train=y_train,
        X_val_text=X_val,
        y_val=y_val,
        n_trials=8,
        random_state=42,
        retrain_on_train_val=True,
        early_stop_patience=3,
        tune_train_sample_ratio=0.7,
        tune_val_sample_ratio=1.0,
        model_filename='XGB_model_custom.pkl',
        vectorizer_filename='XGB_vectorizer.pkl'
    )

    print('Train xong')
    print('Training time:', model_metadata['training_time_sec'], 'giây')
    print('Best params:', model_metadata['best_params'])
    print('Best val score:', model_metadata['best_score_val'])
