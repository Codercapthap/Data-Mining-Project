from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import flask
from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Hiển thị dữ liệu

"""# 1. Đọc dữ liệu"""
data_path = "bank-full.csv"  # load dataset
data = pd.DataFrame(pd.read_csv(data_path, delimiter=";"))
print(data)

print(data.describe())

# job
# marital
# education
# default (binary)
# housing (binary)
# loan (binary)
# contact
# month
# poutcome

ohejob = preprocessing.OneHotEncoder()
X = ohejob.fit_transform(data.job.values.reshape(-1, 1)).toarray()
OHEjob = pd.DataFrame(X, columns=["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                                  "blue-collar", "self-employed", "retired", "technician", "services"])
data = pd.concat([data, OHEjob], axis=1)
data = data.drop("job", axis=1)

ohemarital = preprocessing.OneHotEncoder()
X = ohemarital.fit_transform(data.marital.values.reshape(-1, 1)).toarray()
OHEmarital = pd.DataFrame(X, columns=["married", "divorced", "single"])
data = pd.concat([data, OHEmarital], axis=1)
data = data.drop("marital", axis=1)

oheeducation = preprocessing.OneHotEncoder()
X = oheeducation.fit_transform(data.education.values.reshape(-1, 1)).toarray()
OHEeducation = pd.DataFrame(
    X, columns=["unknown", "secondary", "primary", "tertiary"])
data = pd.concat([data, OHEeducation], axis=1)
data = data.drop("education", axis=1)

ohecontact = preprocessing.OneHotEncoder()
X = ohecontact.fit_transform(data.contact.values.reshape(-1, 1)).toarray()
OHEcontact = pd.DataFrame(X, columns=["unknown", "telephone", "cellular"])
data = pd.concat([data, OHEcontact], axis=1)
data = data.drop("contact", axis=1)

ohemonth = preprocessing.OneHotEncoder()
X = ohemonth.fit_transform(data.month.values.reshape(-1, 1)).toarray()
OHEmonth = pd.DataFrame(X, columns=[
                        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
data = pd.concat([data, OHEmonth], axis=1)
data = data.drop("month", axis=1)

ohepoutcome = preprocessing.OneHotEncoder()
X = ohepoutcome.fit_transform(data.poutcome.values.reshape(-1, 1)).toarray()
OHEpoutcome = pd.DataFrame(
    X, columns=["unknown", "other", "failure", "success"])
data = pd.concat([data, OHEpoutcome], axis=1)
data = data.drop("poutcome", axis=1)

ledefault = preprocessing.LabelEncoder()
data["default"] = ledefault.fit_transform(data['default'])

lehousing = preprocessing.LabelEncoder()
data["housing"] = lehousing.fit_transform(data['housing'])

leloan = preprocessing.LabelEncoder()
data["loan"] = leloan.fit_transform(data['loan'])


ley = preprocessing.LabelEncoder()
data["y"] = ley.fit_transform(data['y'])
data = data.drop("unknown", axis=1)

X = data.iloc[:, data.columns != 'y']
y = data['y']
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)


@app.route('/api/getResult', methods=['POST'])
def api_all():

    content = request.json
    age = content['age']
    job = content['job']
    marital = content['marital']
    education = content['education']
    default = content['default']
    balance = content['balance']
    housing = content['housing']
    loan = content['loan']
    contact = content['contact']
    day = content['day']
    month = content['month']
    duration = content['duration']
    campaign = content['campaign']
    pdays = content['pdays']
    previous = content['previous']
    poutcome = content['poutcome']

    df = pd.DataFrame({
        "age": [age, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "job": [job, "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                "blue-collar", "self-employed", "retired", "technician", "services"],
        "marital": [marital, "married", "divorced", "single", "married", "divorced", "single", "married", "divorced", "single", "married", "divorced", "single"],
        "education": [education, "unknown", "secondary", "primary", "tertiary", "unknown", "secondary", "primary", "tertiary", "unknown", "secondary", "primary", "tertiary"],
        "default": [default, "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        "balance": [balance, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "housing": [housing, "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        "loan": [loan, "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        "contact": [contact, "unknown", "telephone", "cellular", "unknown", "telephone", "cellular", "unknown", "telephone", "cellular", "unknown", "telephone", "cellular"],
        "day": [day, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "month": [month, "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "duration": [duration, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "campaign": [campaign, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "pdays": [pdays, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "previous": [previous, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "poutcome": [poutcome, "unknown", "other", "failure", "success", "unknown", "other", "failure", "success", "unknown", "other", "failure", "success"]
    })

    ohejob = preprocessing.OneHotEncoder()
    X = ohejob.fit_transform(df.job.values.reshape(-1, 1)).toarray()
    OHEjob = pd.DataFrame(X, columns=["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                                      "blue-collar", "self-employed", "retired", "technician", "services"])
    df = pd.concat([df, OHEjob], axis=1)
    df = df.drop("job", axis=1)

    ohemarital = preprocessing.OneHotEncoder()
    X = ohemarital.fit_transform(df.marital.values.reshape(-1, 1)).toarray()
    OHEmarital = pd.DataFrame(X, columns=["married", "divorced", "single"])
    df = pd.concat([df, OHEmarital], axis=1)
    df = df.drop("marital", axis=1)

    oheeducation = preprocessing.OneHotEncoder()
    X = oheeducation.fit_transform(
        df.education.values.reshape(-1, 1)).toarray()
    OHEeducation = pd.DataFrame(
        X, columns=["unknown", "secondary", "primary", "tertiary"])
    df = pd.concat([df, OHEeducation], axis=1)
    df = df.drop("education", axis=1)

    ohecontact = preprocessing.OneHotEncoder()
    X = ohecontact.fit_transform(df.contact.values.reshape(-1, 1)).toarray()
    OHEcontact = pd.DataFrame(X, columns=["unknown", "telephone", "cellular"])
    df = pd.concat([df, OHEcontact], axis=1)
    df = df.drop("contact", axis=1)

    ohemonth = preprocessing.OneHotEncoder()
    X = ohemonth.fit_transform(df.month.values.reshape(-1, 1)).toarray()
    OHEmonth = pd.DataFrame(X, columns=[
                            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    df = pd.concat([df, OHEmonth], axis=1)
    df = df.drop("month", axis=1)

    ohepoutcome = preprocessing.OneHotEncoder()
    X = ohepoutcome.fit_transform(df.poutcome.values.reshape(-1, 1)).toarray()
    OHEpoutcome = pd.DataFrame(
        X, columns=["unknown", "other", "failure", "success"])
    df = pd.concat([df, OHEpoutcome], axis=1)
    df = df.drop("poutcome", axis=1)
    ledefault = preprocessing.LabelEncoder()
    df["default"] = ledefault.fit_transform(df['default'])

    lehousing = preprocessing.LabelEncoder()
    df["housing"] = lehousing.fit_transform(df['housing'])

    leloan = preprocessing.LabelEncoder()
    df["loan"] = leloan.fit_transform(df['loan'])
    df = df.drop("unknown", axis=1)

    print(df)
    print(df.head(1))

    prediction = model.predict(df.head(1))
    result = prediction.tolist()
    return flask.jsonify(result)


# [{"age": age, "job": job, "marital": marital, "education": education, "default": default, "balance": balance, "housing": housing,
#   "loan": loan, "contact": contact, "day": day, "month": month, "duration": duration, "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome}]
app.run()
