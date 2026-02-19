from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from calidad_pqrs.utils import save_model
from calidad_pqrs.config import MODEL_CAUSES_DIR



def tune_causes_model(dataset, n_trials):

    X_train, X_test, y_train, y_test = train_test_split(
        dataset[['Descripción_TfIdf', 'Proceso', 'Filtro 4_map', 'Filtro 3_map']],
        dataset['Causa'],
        test_size = 0.2,
        shuffle = True,
        random_state= 666,
    )

    def objective(trial):

        min_df = trial.suggest_float('min_df', 1e-5, 9e-4)
        C = trial.suggest_float('C', 9e-1, 1e1)
        percentile = trial.suggest_int('percentile', 30, 90)

        preprocessor = ColumnTransformer(
            transformers = [
                ('tfidf', TfidfVectorizer(decode_error='ignore', min_df=min_df, max_df=0.8), 'Descripción_TfIdf'),
                ('ohe', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), ['Proceso',  'Filtro 4_map', 'Filtro 3_map']),
            ]
        )

        pipe = Pipeline(
            [
                ('preprocessor', preprocessor),
                ('select', SelectPercentile(score_func=chi2, percentile=percentile)),
                ('classifier', LogisticRegression(random_state=666, max_iter=7000, class_weight='balanced', penalty='l2', solver='liblinear', C=C))
            ]
        )

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=666)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro')

        return scores.mean()


    study = optuna.create_study(direction="maximize", study_name='logistic_regression_optimization')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params


    print('\nEntrenando regresión logística con los mejores hiperparámetros encontrados...')
    preprocessor = ColumnTransformer(
        transformers = [
            ('tfidf', TfidfVectorizer(decode_error='ignore', min_df=best_params['min_df'], max_df=0.8), 'Descripción_TfIdf'),
            ('ohe', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), ['Proceso', 'Filtro 4_map', 'Filtro 3_map']),
        ]
    )

    model_causes = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('select', SelectPercentile(score_func=chi2, percentile=best_params['percentile'])),
            ('classifier', LogisticRegression(random_state=666, max_iter=7000, class_weight='balanced', penalty='l2', solver='liblinear', C=best_params['C']))
        ]
    )

    model_causes.fit(X_train, y_train)


    save_model(
        model = model_causes,
        model_dir = MODEL_CAUSES_DIR,
        model_name = 'salud_causes_classifier.pkl',
        X_test = X_test,
        y_test = y_test
    )

    return X_test, y_test
