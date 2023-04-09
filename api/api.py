from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import flask
from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Hiển thị dữ liệu

"""# 1. Đọc dữ liệu"""
data_path = "bank-full.csv"  # load dataset
data = pd.DataFrame(pd.read_csv(data_path, delimiter=";"))

"""# 2. Xử lí dữ liệu"""
# chia các dữ liệu chữ thành các cột
OHE = preprocessing.OneHotEncoder()
X = OHE.fit_transform(data.job.values.reshape(-1, 1)).toarray()
OHEjob = pd.DataFrame(X, columns=["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                                  "blue-collar", "self-employed", "retired", "technician", "services"])
data = pd.concat([data, OHEjob], axis=1)
data = data.drop("job", axis=1)

X = OHE.fit_transform(data.marital.values.reshape(-1, 1)).toarray()
OHEmarital = pd.DataFrame(X, columns=["married", "divorced", "single"])
data = pd.concat([data, OHEmarital], axis=1)
data = data.drop("marital", axis=1)

X = OHE.fit_transform(data.education.values.reshape(-1, 1)).toarray()
OHEeducation = pd.DataFrame(
    X, columns=["unknown", "secondary", "primary", "tertiary"])
data = pd.concat([data, OHEeducation], axis=1)
data = data.drop("education", axis=1)

X = OHE.fit_transform(data.contact.values.reshape(-1, 1)).toarray()
OHEcontact = pd.DataFrame(X, columns=["unknown", "telephone", "cellular"])
data = pd.concat([data, OHEcontact], axis=1)
data = data.drop("contact", axis=1)

X = OHE.fit_transform(data.month.values.reshape(-1, 1)).toarray()
OHEmonth = pd.DataFrame(X, columns=[
                        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
data = pd.concat([data, OHEmonth], axis=1)
data = data.drop("month", axis=1)

X = OHE.fit_transform(data.poutcome.values.reshape(-1, 1)).toarray()
OHEpoutcome = pd.DataFrame(
    X, columns=["unknown", "other", "failure", "success"])
data = pd.concat([data, OHEpoutcome], axis=1)
data = data.drop("poutcome", axis=1)

# dữ liệu yes no số hóa thành 1 và 0
labelEncode = preprocessing.LabelEncoder()
data["default"] = labelEncode.fit_transform(data['default'])

data["housing"] = labelEncode.fit_transform(data['housing'])

data["loan"] = labelEncode.fit_transform(data['loan'])

data["y"] = labelEncode.fit_transform(data['y'])

# bỏ đi cột unknown vì không có giá trị thực tế
data = data.drop("unknown", axis=1)

# chia dữ liệu thành tập thuộc tính và tập nhãn
X = data.iloc[:, data.columns != 'y']
y = data['y']

"""# 3. Huấn luyện mô hình"""
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
