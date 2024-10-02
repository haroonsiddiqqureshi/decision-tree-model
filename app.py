import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request
from config import (
    APP_HOST,
    APP_DEBUG_MODE,
    APP_PORT,
    APP_TEMPLATE_FOLDER,
    APP_STATIC_FOLDER,
    APP_STATIC_URL_PATH,
)

app = Flask(
    __name__,
    template_folder=APP_TEMPLATE_FOLDER,
    static_folder=APP_STATIC_FOLDER,
    static_url_path=APP_STATIC_URL_PATH,
)

CSV_FILE = "static/data/diabetes.csv"
DIABETES_DATA = pd.read_csv(CSV_FILE)

DIABETES_COLUMNS = {
    "Pregnancies": "จำนวนการตั้งครรภ์",
    "Glucose": "ระดับกลูโคสในเลือด",
    "BloodPressure": "ความดันโลหิต",
    "SkinThickness": "ความหนาของผิวหนัง",
    "Insulin": "ระดับอินซูลิน",
    "BMI": "ดัชนีมวลกาย",
    "DiabetesPedigreeFunction": "คะแนนความเสี่ยงโรคเบาหวานจากประวัติครอบครัว",
    "Age": "อายุ",
    "Outcome": "สถานะการเกิดโรคเบาหวาน (1 = มีโรคเบาหวาน; 0 = ไม่มีโรคเบาหวาน)",
}

SELECTED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@app.route("/", methods=["GET"])
def table():
    diabetes_table = DIABETES_DATA.head(50).to_dict(orient="records")
    return render_template(
        "table.html",
        diabetes_table=diabetes_table,
        DIABETES_COLUMNS=DIABETES_COLUMNS,
    )


@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict.html")


@app.route("/predict/result", methods=["POST"])
def predict_result():
    result_predict = None

    # เตรียมข้อมูลตัวแปร X และ Outcome
    X = DIABETES_DATA[SELECTED_FEATURES]
    y = DIABETES_DATA["Outcome"]

    # แบ่งข้อมูลออกเป็น train และ test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # สร้างและฝึกโมเดล
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # รับข้อมูลที่ผู้ใช้กรอกในฟอร์ม
    user_input = {
        "Pregnancies": float(request.form["pregnancies"]),
        "Glucose": float(request.form["glucose"]),
        "BloodPressure": float(request.form["bloodpressure"]),
        "SkinThickness": float(request.form["skinthickness"]),
        "Insulin": float(request.form["insulin"]),
        "BMI": float(request.form["bmi"]),
        "DiabetesPedigreeFunction": float(request.form["diabetespedigreefunction"]),
        "Age": float(request.form["age"]),
    }

    # แปลงข้อมูลของผู้ใช้ (ค่าที่อยู่ใน dictionary) เป็นอาร์เรย์ของ NumPy
    # ปรับรูปร่างของอาร์เรย์ให้มี 1 แถวและจำนวนคอลัมน์เท่ากับจำนวนฟีเจอร์ (-1 ทำให้ NumPy คำนวณจำนวนคอลัมน์เอง)
    user_input_array = np.array(list(user_input.values())).reshape(1, -1)

    # ใช้โมเดลที่ฝึกแล้วเพื่อทำนายว่าข้อมูลที่ผู้ใช้กรอกบ่งบอกถึงการเป็นโรคหัวใจ (1) หรือไม่ (0)
    prediction = model.predict(user_input_array)

    # กำหนดข้อความผลลัพธ์ตามการทำนาย:
    # หากการทำนายเป็น 1 หมายถึงผู้ใช้เป็นโรคเบาหวาน, หากเป็น 0 หมายถึงไม่เป็นโรคเบาหวาน
    result_predict = "เป็นโรคเบาหวาน" if prediction[0] == 1 else "ไม่เป็นโรคเบาหวาน"

    return render_template("predict_result.html", result_predict=result_predict)


if __name__ == "__main__":
    app.run(host=APP_HOST, debug=APP_DEBUG_MODE, port=APP_PORT)
