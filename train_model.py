
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score

CSV_PATH      = "training_data.csv"
MODEL_PATH    = "cheat_model.pkl"
SCALER_PATH   = "cheat_scaler.pkl"
FEATURES_PATH = "cheat_features.pkl"

FEATURES = [
    "yaw", "pitch", "iris",
    "face_visible", "tab_switch",
    "iris_dev", "iris_from_left", "iris_from_right", "iris_out", "iris_sq",
    "yaw_dev", "pitch_dev", "yaw_out", "pitch_out", "yaw_sq",
    "iris_x_yaw", "combined_dev", "away_score"
]


def engineer(df):
    df = df.copy()
    df["iris"]         = df["iris"].astype(float)
    df["yaw"]          = df["yaw"].astype(float)
    df["pitch"]        = df["pitch"].astype(float)
    df["face_visible"] = df["face_visible"].astype(int)
    df["tab_switch"]   = df["tab_switch"].astype(int)

    df["iris_dev"]        = (df["iris"]  - 0.20).abs()
    df["iris_from_left"]  =  df["iris"]  - 0.09
    df["iris_from_right"] =  0.31 - df["iris"]
    df["iris_out"]        = ((df["iris"] < 0.09) | (df["iris"] > 0.31)).astype(int)
    df["iris_sq"]         =  df["iris_dev"] ** 2
    df["yaw_dev"]         = (df["yaw"]   - 26.0).abs()
    df["pitch_dev"]       = (df["pitch"] - 17.0).abs()
    df["yaw_out"]         = ((df["yaw"]  < 18)  | (df["yaw"]  > 34)).astype(int)
    df["pitch_out"]       = ((df["pitch"]< 10)  | (df["pitch"]> 24)).astype(int)
    df["yaw_sq"]          =  df["yaw_dev"] ** 2
    df["iris_x_yaw"]      =  df["iris_dev"] * df["yaw_dev"]
    df["combined_dev"]    =  df["iris_dev"]*2 + df["yaw_dev"]*0.1 + df["pitch_dev"]*0.05
    df["away_score"]      = (1 - df["face_visible"])*10 + df["tab_switch"]*10
    return df


def main():
    print("\n" + "="*52)
    print("  AI Proctor — Random Forest Training")
    print("="*52)

    # ── Load ──
    if not os.path.exists(CSV_PATH):
        print(f"\n[ERROR] {CSV_PATH} not found.")
        print("  → Run exam_camera_detection.py first, then press ESC.")
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df["label"].isin(["cheating","normal"])].drop_duplicates().dropna()

    print(f"\n  CSV rows loaded : {len(df)}")
    print(f"  Normal          : {(df.label=='normal').sum()}")
    print(f"  Cheating        : {(df.label=='cheating').sum()}")

    if len(df) < 100:
        print(f"\n[ERROR] Only {len(df)} samples. Need at least 100.")
        print("  → Run more exam sessions to collect data, then retrain.")
        return

    # ── Feature engineering ──
    df = engineer(df)
    X  = df[FEATURES].values
    y  = (df["label"] == "cheating").astype(int).values

    # ── Train / test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler    = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # ── Train ──
    print("\n  Training Random Forest (500 trees)... please wait")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(Xtr, y_train)

    # ── Evaluate ──
    y_pred = rf.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  ✅ Test Accuracy    : {acc*100:.2f}%")

    cv = cross_val_score(rf, Xtr, y_train, cv=5, scoring="accuracy")
    print(f"  5-Fold CV          : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

    print("\n  --- Classification Report ---")
    print(classification_report(y_test, y_pred,
                                target_names=["normal","cheating"],
                                digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print("  --- Confusion Matrix ---")
    print("                Pred:Normal   Pred:Cheating")
    print(f"  Act:Normal       {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Act:Cheating     {cm[1][0]:>5}          {cm[1][1]:>5}")
    print()
    print(f"  True Negatives  (Normal  correctly normal)   : {cm[0][0]}")
    print(f"  False Positives (Normal  wrongly cheating)   : {cm[0][1]}")
    print(f"  False Negatives (Cheating wrongly normal)    : {cm[1][0]}")
    print(f"  True Positives  (Cheating correctly caught)  : {cm[1][1]}")

    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n  --- Top 8 Features ---")
    for feat, val in fi.head(8).items():
        bar = "█" * int(val * 200)
        print(f"  {feat:<22} {val:.4f}  {bar}")

    print()
    if acc >= 0.95:
        print(f"  ✅ EXCELLENT — {acc*100:.1f}% accuracy. Model is production ready!")
    elif acc >= 0.82:
        print(f"  ✅ GOOD — {acc*100:.1f}% accuracy. Above the 82% threshold.")
    else:
        print(f"  ⚠  {acc*100:.1f}% — Below 82%. Tips to improve:")
        print("     1. Collect more sessions (500+ rows per class)")
        print("     2. Record in different lighting conditions")
        print("     3. Vary your distance (50–70 cm)")
        print("     4. Use manual C/N labelling for edge cases")

    # ── Save ──
    joblib.dump(rf,       MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)
    print(f"\n  Saved: {MODEL_PATH}")
    print(f"  Saved: {SCALER_PATH}")
    print(f"  Saved: {FEATURES_PATH}")
    print("="*52 + "\n")


if __name__ == "__main__":
    main()
