from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from calidad_pqrs.utils import save_model
from calidad_pqrs.config import MODEL_PROCESS_DIR



def tune_process_model(dataset, n_trials):

    X_train, X_test, y_train, y_test = train_test_split(
        dataset['Descripción_TfIdf'],
        dataset['Proceso'],
        test_size = 0.2,
        shuffle = True,
        random_state= 666,
    )


    def objective(trial):

        percentile = trial.suggest_int('percentile', 40, 100)
        C = trial.suggest_float('C', 1e-1, 1e1, log=True)
        min_df = trial.suggest_float('min_df', 1e-5, 9e-4)

        pipe = Pipeline(
            [
                ('preprocessor', TfidfVectorizer(decode_error='ignore', min_df=min_df, max_df=0.8)),
                ('select', SelectPercentile(score_func=chi2, percentile=percentile)),
                ('classifier', LogisticRegression(
                    random_state=666,
                    max_iter=6000,
                    class_weight='balanced',
                    penalty='l2',
                    solver='liblinear',
                    C=C
                )),
            ]
        )

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=666)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)

        return scores.mean()


    study = optuna.create_study(direction="maximize", study_name='logistic_regression_optimization')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params


    print('\nEntrenando regresión logística con los mejores hiperparámetros encontrados...')
    model_process = Pipeline(
        [
            ('preprocessor', TfidfVectorizer(decode_error='ignore', min_df=best_params['min_df'], max_df=0.8)),
            ('select', SelectPercentile(score_func=chi2, percentile=best_params['percentile'])),
            ('classifier', LogisticRegression(random_state=666, max_iter=6000, class_weight='balanced', penalty='l2', solver='liblinear', C=best_params['C'])),
        ]
    )

    model_process.fit(X_train, y_train)


    save_model(
        model = model_process,
        model_dir = MODEL_PROCESS_DIR,
        model_name = 'salud_process_classifier.pkl',
        X_test = X_test,
        y_test = y_test
    )

    return X_test, y_test