from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random


model_dir = Path.cwd().joinpath('model')
model_dir.mkdir(exist_ok=True)
MODEL_PATH = model_dir.joinpath('model.json')

def read_features(all_data):
    
    all_data['WIDTH'] = all_data["WIDTH"]
    

    return all_data

def train_model(all_data):
    X, y = all_data[['WIDTH']], all_data["IS_LICENSE_PLATE"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 1000))

    # fit model no training data
    model = XGBClassifier(n_estimators=5000)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50)

    print(accuracy_score(y_train, model.predict(X_train)), accuracy_score(y_test, model.predict(X_test)))

    all_data["MODEL_RESULT"] = model.predict(X)
    all_data.to_csv(Path.cwd().joinpath('xgboost_results.csv'))

    return model

def get_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    return model