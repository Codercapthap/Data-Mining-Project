import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd

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

# cân bằng dữ liệu
count_0, count_1 = data.y.value_counts()
data_0 = data[data['y'] == 0]
data_1 = data[data['y'] == 1]
data_0_under = data_0.sample(count_1)
data = pd.concat([data_0_under, data_1])

# chia dữ liệu thành tập thuộc tính và tập nhãn
X = data.iloc[:, data.columns != 'y']
y = data['y']

"""# 3. Huấn luyện mô hình"""
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the model
joblib.dump(model, "my_model.pkl")
