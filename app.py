from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from email_validator import validate_email, EmailNotValidError
import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
app.secret_key = "your_secret_key_12345"

DATABASE = "users.db"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
FIXED_UPLOAD_FILE = "uploaded_dataset.csv"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT NOT NULL,
            mobile TEXT NOT NULL,
            address TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            speed REAL,
            road_condition_status TEXT,
            prediction_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_uploaded_dataset_path():
    return os.path.join(UPLOAD_FOLDER, FIXED_UPLOAD_FILE)


def preprocess_dataframe(df):
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    df[col] = df[col].fillna("Unknown")
        else:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())

    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        email = request.form.get("email", "").strip()
        mobile = request.form.get("mobile", "").strip()
        address = request.form.get("address", "").strip()

        if not username or not password or not email or not mobile or not address:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "danger")
            return redirect(url_for("register"))

        if len(password) < 4:
            flash("Password must be at least 4 characters.", "danger")
            return redirect(url_for("register"))

        if not mobile.isdigit() or len(mobile) != 10:
            flash("Mobile number must be exactly 10 digits.", "danger")
            return redirect(url_for("register"))

        try:
            valid = validate_email(email)
            email = valid.email
        except EmailNotValidError:
            flash("Invalid email address.", "danger")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users (username, password, email, mobile, address)
                VALUES (?, ?, ?, ?, ?)
            """, (username, hashed_password, email, mobile, address))
            conn.commit()
            conn.close()

            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("Username already exists. Please choose another.", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("login"))

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Login successful.", "success")
            return redirect(url_for("user_home"))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/user_home")
def user_home():
    if "username" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    return render_template("user_home.html", username=session["username"])


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin"] = ADMIN_USERNAME
            flash("Admin login successful.", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid admin credentials.", "danger")
            return render_template("admin_login.html")

    return render_template("admin_login.html")


@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session:
        flash("Please login as admin first.", "warning")
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users ORDER BY id DESC")
    users = cur.fetchall()
    conn.close()

    return render_template("admin_dashboard.html", users=users)


@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if "admin" not in session:
        flash("Please login as admin first.", "warning")
        return redirect(url_for("admin_login"))

    columns = []
    preview_data = []
    total_rows = 0
    total_columns = 0
    filename = ""

    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or file.filename == "":
            flash("Please choose a CSV file.", "danger")
            return redirect(url_for("upload_dataset"))

        if not allowed_file(file.filename):
            flash("Only CSV files are allowed.", "danger")
            return redirect(url_for("upload_dataset"))

        filename = secure_filename(file.filename)
        filepath = get_uploaded_dataset_path()
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)

            if df.empty:
                flash("Uploaded dataset is empty.", "danger")
                return redirect(url_for("upload_dataset"))

            total_rows = len(df)
            total_columns = len(df.columns)
            columns = df.columns.tolist()

            sample_size = 10 if len(df) >= 10 else len(df)
            random_df = df.sample(n=sample_size, random_state=None)
            preview_data = random_df.fillna("").to_dict(orient="records")

            flash("Dataset uploaded successfully.", "success")

        except Exception as e:
            flash(f"Error while reading dataset: {str(e)}", "danger")
            return redirect(url_for("upload_dataset"))

    return render_template(
        "upload_dataset.html",
        columns=columns,
        preview_data=preview_data,
        total_rows=total_rows,
        total_columns=total_columns,
        filename=filename
    )


@app.route("/preprocess_dataset")
def preprocess_dataset():
    if "admin" not in session:
        flash("Please login as admin first.", "warning")
        return redirect(url_for("admin_login"))

    filepath = get_uploaded_dataset_path()

    if not os.path.exists(filepath):
        flash("No dataset found. Please upload dataset first.", "danger")
        return redirect(url_for("upload_dataset"))

    try:
        df = pd.read_csv(filepath)

        original_rows = len(df)
        original_columns = len(df.columns)

        missing_before = int(df.isnull().sum().sum())
        duplicate_before = int(df.duplicated().sum())

        processed_df = preprocess_dataframe(df)

        missing_after = int(processed_df.isnull().sum().sum())
        duplicate_after = int(processed_df.duplicated().sum())

        sample_size = 10 if len(processed_df) >= 10 else len(processed_df)
        preview_data = processed_df.sample(n=sample_size).fillna("").to_dict(orient="records")
        columns = processed_df.columns.tolist()

        return render_template(
            "preprocess_dataset.html",
            filename=FIXED_UPLOAD_FILE,
            new_filename="Not Saved (Preview Only)",
            original_rows=original_rows,
            original_columns=original_columns,
            missing_before=missing_before,
            missing_after=missing_after,
            duplicate_before=duplicate_before,
            duplicate_after=duplicate_after,
            columns=columns,
            preview_data=preview_data
        )

    except Exception as e:
        flash(f"Error during preprocessing: {str(e)}", "danger")
        return redirect(url_for("admin_dashboard"))


@app.route("/train_model")
def train_model():
    if "admin" not in session:
        flash("Please login as admin first.", "warning")
        return redirect(url_for("admin_login"))

    filepath = get_uploaded_dataset_path()

    if not os.path.exists(filepath):
        flash("Please upload dataset first.", "danger")
        return redirect(url_for("upload_dataset"))

    try:
        df = pd.read_csv(filepath)

        if df.empty:
            flash("Dataset is empty.", "danger")
            return redirect(url_for("upload_dataset"))

        df = preprocess_dataframe(df)

        target_column = None
        for col in df.columns:
            if col.strip().lower() in ["accident", "target", "label", "class"]:
                target_column = col
                break

        if target_column is None:
            flash("Target column not found in dataset.", "danger")
            return redirect(url_for("preprocess_dataset"))

        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        for col in X.columns:
            if X[col].dtype == "object":
                le_x = LabelEncoder()
                X[col] = le_x.fit_transform(X[col].astype(str))

        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y.astype(str))
        class_names = list(y_encoder.classes_)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            flash("Training needs at least 2 target classes.", "danger")
            return redirect(url_for("preprocess_dataset"))

        if np.min(class_counts) < 2:
            flash("Each class must have at least 2 records for train/test split.", "danger")
            return redirect(url_for("preprocess_dataset"))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        results = []
        model_names = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        confusion_images = {}

        plot_folder = os.path.join("static", "plots")
        os.makedirs(plot_folder, exist_ok=True)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append({
                "name": model_name,
                "accuracy": round(acc, 4),
                "precision": round(pre, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4)
            })

            model_names.append(model_name)
            accuracy_scores.append(acc)
            precision_scores.append(pre)
            recall_scores.append(rec)
            f1_scores.append(f1)

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")

            ax.set_title(f"{model_name} Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold"
                    )

            fig.colorbar(im)
            plt.tight_layout()

            cm_filename = model_name.lower().replace(" ", "_") + "_cm.png"
            cm_path = os.path.join(plot_folder, cm_filename)
            plt.savefig(cm_path, bbox_inches="tight")
            plt.close()

            confusion_images[model_name] = cm_path.replace("\\", "/")

        x = np.arange(len(model_names))
        width = 0.2

        plt.figure(figsize=(10, 6))
        plt.bar(x - 1.5 * width, accuracy_scores, width, label="Accuracy", color="blue")
        plt.bar(x - 0.5 * width, precision_scores, width, label="Precision", color="green")
        plt.bar(x + 0.5 * width, recall_scores, width, label="Recall", color="orange")
        plt.bar(x + 1.5 * width, f1_scores, width, label="F1 Score", color="red")

        plt.xticks(x, model_names)
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("Algorithm Performance Comparison")
        plt.legend()
        plt.tight_layout()

        combined_graph = os.path.join(plot_folder, "combined_metrics.png")
        plt.savefig(combined_graph, bbox_inches="tight")
        plt.close()

        return render_template(
            "train_model.html",
            results=results,
            combined_graph=combined_graph.replace("\\", "/"),
            confusion_images=confusion_images
        )

    except Exception as e:
        flash(f"Error during model training: {str(e)}", "danger")
        return redirect(url_for("admin_dashboard"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "username" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))

    filepath = get_uploaded_dataset_path()

    if not os.path.exists(filepath):
        flash("Dataset not found. Admin must upload dataset first.", "danger")
        return redirect(url_for("user_home"))

    try:
        df = pd.read_csv(filepath)

        if df.empty:
            flash("Dataset is empty.", "danger")
            return redirect(url_for("user_home"))

        df = preprocess_dataframe(df)

        target_column = None
        for col in df.columns:
            if col.strip().lower() in ["accident", "target", "label", "class"]:
                target_column = col
                break

        if target_column is None:
            flash("Target column not found in dataset.", "danger")
            return redirect(url_for("user_home"))

        input_columns = [col for col in df.columns if col != target_column]

        prediction_result = None
        road_condition_status = None
        form_values = {}
        dropdown_options = {}
        manual_input_columns = []

        for col in input_columns:
            if col.strip().lower() == "speed":
                manual_input_columns.append(col)
            elif df[col].dtype == "object":
                dropdown_options[col] = sorted(df[col].astype(str).dropna().unique().tolist())

        if request.method == "POST":
            X = df[input_columns].copy()
            y = df[target_column].copy()

            encoders = {}

            for col in X.columns:
                if X[col].dtype == "object":
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le

            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y.astype(str))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)

            user_row = []

            for col in input_columns:
                value = request.form.get(col, "").strip()
                form_values[col] = value

                if col.strip().lower() == "speed":
                    try:
                        numeric_value = float(value)
                    except:
                        numeric_value = float(df[col].mean())
                    user_row.append(numeric_value)

                elif df[col].dtype == "object":
                    le = encoders[col]
                    known_classes = list(le.classes_)

                    if value not in known_classes:
                        value = known_classes[0]

                    encoded_value = le.transform([value])[0]
                    user_row.append(encoded_value)

                else:
                    numeric_value = float(df[col].mean())
                    user_row.append(numeric_value)

            user_row = np.array(user_row).reshape(1, -1)
            user_row_scaled = scaler.transform(user_row)

            pred = model.predict(user_row_scaled)[0]
            prediction_result = y_encoder.inverse_transform([pred])[0]

            road_condition_status = "Good"

            road_condition_value = str(request.form.get("Road_Condition", "")).strip().lower()
            visibility_value = str(request.form.get("Visibility", "")).strip().lower()
            weather_value = str(request.form.get("Weather", "")).strip().lower()

            if road_condition_value in ["poor", "bad", "wet", "damaged", "slippery", "muddy"]:
                road_condition_status = "Risky"
            elif visibility_value in ["low", "poor", "foggy", "very low"]:
                road_condition_status = "Risky"
            elif weather_value in ["rainy", "stormy", "foggy", "snowy"]:
                road_condition_status = "Risky"

            speed_value = request.form.get("Speed", "").strip()
            try:
                speed_value = float(speed_value)
            except:
                speed_value = 0

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO prediction_history (username, speed, road_condition_status, prediction_result)
                VALUES (?, ?, ?, ?)
            """, (
                session["username"],
                speed_value,
                road_condition_status,
                str(prediction_result)
            ))
            conn.commit()
            conn.close()

            return render_template(
                "predict.html",
                input_columns=input_columns,
                manual_input_columns=manual_input_columns,
                dropdown_options=dropdown_options,
                prediction_result=prediction_result,
                road_condition_status=road_condition_status,
                form_values=form_values
            )

        return render_template(
            "predict.html",
            input_columns=input_columns,
            manual_input_columns=manual_input_columns,
            dropdown_options=dropdown_options,
            prediction_result=prediction_result,
            road_condition_status=road_condition_status,
            form_values=form_values
        )

    except Exception as e:
        flash(f"Prediction error: {str(e)}", "danger")
        return redirect(url_for("user_home"))


@app.route("/prediction_history")
def prediction_history():
    if "username" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, speed, road_condition_status, prediction_result, created_at
        FROM prediction_history
        WHERE username = ?
        ORDER BY id DESC
    """, (session["username"],))
    history_data = cur.fetchall()
    conn.close()

    return render_template("prediction_history.html", history_data=history_data)


@app.route("/delete_prediction_history/<int:history_id>")
def delete_prediction_history(history_id):
    if "username" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM prediction_history
        WHERE id = ? AND username = ?
    """, (history_id, session["username"]))
    conn.commit()
    conn.close()

    flash("Prediction history deleted successfully.", "success")
    return redirect(url_for("prediction_history"))


@app.route("/admin_prediction_history")
def admin_prediction_history():
    if "admin" not in session:
        flash("Please login as admin first.", "warning")
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, username, speed, road_condition_status, prediction_result, created_at
        FROM prediction_history
        ORDER BY id DESC
    """)

    history_data = cur.fetchall()
    conn.close()

    return render_template(
        "admin_prediction_history.html",
        history_data=history_data
    )

@app.route("/admin_delete_history/<int:history_id>")
def admin_delete_history(history_id):

    if "admin" not in session:
        flash("Please login as admin first.", "warning")
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM prediction_history WHERE id = ?",
        (history_id,)
    )

    conn.commit()
    conn.close()

    flash("History deleted successfully", "success")

    return redirect(url_for("admin_prediction_history"))    


@app.route("/admin_logout")
def admin_logout():
    session.pop("admin", None)
    flash("Admin logged out successfully.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)