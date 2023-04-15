import pandas as pd
import flask
from flask import request
from flask_cors import CORS, cross_origin
import joblib
from sklearn import preprocessing

app = flask.Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the saved model
model = joblib.load('my_model.pkl')
OHE = preprocessing.OneHotEncoder()
labelEncode = preprocessing.LabelEncoder()


@app.route('/api/getResult', methods=['POST'])
@cross_origin()
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

    """# Xử lí dữ liệu input"""
    dfInput = pd.DataFrame({
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

    X = OHE.fit_transform(dfInput.job.values.reshape(-1, 1)).toarray()
    OHEjob = pd.DataFrame(X, columns=["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                                      "blue-collar", "self-employed", "retired", "technician", "services"])
    dfInput = pd.concat([dfInput, OHEjob], axis=1)
    dfInput = dfInput.drop("job", axis=1)

    X = OHE.fit_transform(dfInput.marital.values.reshape(-1, 1)).toarray()
    OHEmarital = pd.DataFrame(X, columns=["married", "divorced", "single"])
    dfInput = pd.concat([dfInput, OHEmarital], axis=1)
    dfInput = dfInput.drop("marital", axis=1)

    X = OHE.fit_transform(
        dfInput.education.values.reshape(-1, 1)).toarray()
    OHEeducation = pd.DataFrame(
        X, columns=["unknown", "secondary", "primary", "tertiary"])
    dfInput = pd.concat([dfInput, OHEeducation], axis=1)
    dfInput = dfInput.drop("education", axis=1)

    X = OHE.fit_transform(dfInput.contact.values.reshape(-1, 1)).toarray()
    OHEcontact = pd.DataFrame(X, columns=["unknown", "telephone", "cellular"])
    dfInput = pd.concat([dfInput, OHEcontact], axis=1)
    dfInput = dfInput.drop("contact", axis=1)

    X = OHE.fit_transform(dfInput.month.values.reshape(-1, 1)).toarray()
    OHEmonth = pd.DataFrame(X, columns=[
                            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    dfInput = pd.concat([dfInput, OHEmonth], axis=1)
    dfInput = dfInput.drop("month", axis=1)

    X = OHE.fit_transform(dfInput.poutcome.values.reshape(-1, 1)).toarray()
    OHEpoutcome = pd.DataFrame(
        X, columns=["unknown", "other", "failure", "success"])
    dfInput = pd.concat([dfInput, OHEpoutcome], axis=1)
    dfInput = dfInput.drop("poutcome", axis=1)

    dfInput["default"] = labelEncode.fit_transform(dfInput['default'])

    dfInput["housing"] = labelEncode.fit_transform(dfInput['housing'])

    dfInput["loan"] = labelEncode.fit_transform(dfInput['loan'])
    dfInput = dfInput.drop("unknown", axis=1)

    """# Dự đoán bằng mô hình và trả về"""
    prediction = model.predict(dfInput.head(1))
    result = prediction.tolist()
    return flask.jsonify(result[0])


app.run()
