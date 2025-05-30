import io
import os

import pickle
from catboost import CatBoostClassifier, CatBoostRegressor

from storage import get_dir


def print_important_words(x, vectorizer):
    scores = x.toarray()[0]

    out = vectorizer.get_feature_names_out()
    important_words = [word for word, score in zip(out, scores) if score >= 0.1]

    for word in important_words:
        print('-', word)


def get_output_dir(language, target, source, classifier):
    return get_dir(language, target, where='log', source=source, classifier=classifier)


def save_model(model, vectorizer, output_dir, s3=None):
    path = f'{output_dir}/model.cbm'
    model.save_model(path)
    if s3:
        with open(path, 'rb') as src:
            s3.write(path, src)
        os.remove(path)

    save_vectorizer(vectorizer, output_dir, s3)


def save_vectorizer(vectorizer, output_dir, s3=None):
    path = f'{output_dir}/vectorizer.pkl'

    if s3:
        buf = io.BytesIO()
        pickle.dump(vectorizer, buf)
        s3.write(path, buf)
    else:
        with open(path, 'wb') as dst:
            pickle.dump(vectorizer, dst)


def get_model(classifier):
    if classifier.get('problem_type') == 'multi_label_classification':
        model = CatBoostRegressor()
    else:
        model = CatBoostClassifier()
    return model


def load_saved_model(classifier, language, target, train, s3=None):
    output_dir = get_output_dir(language, target, train, classifier['classifier'])

    model = get_model(classifier)

    model_path = f'{output_dir}/model.cbm'
    vectorizer_path = f'{output_dir}/vectorizer.pkl'

    if s3 is not None:
        with s3.open(model_path, mode='b') as src:
            model.load_model(stream=src)
        with s3.open(vectorizer_path, mode='b') as src:
            vectorizer = pickle.load(src)
    else:
        model.load_model(f'{output_dir}/model.cbm')
        with open(f'{output_dir}/vectorizer.pkl', 'rb') as src:
            vectorizer = pickle.load(src)

    return model, vectorizer
