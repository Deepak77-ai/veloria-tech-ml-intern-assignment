# model.py
# Task 2 - Build a machine learning model to predict match winners
# I tried multiple algorithms and picked the best one automatically

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

print("=" * 50)
print("Task 2 - Match Winner Prediction Model")
print("=" * 50)


# Step 1 - Load the CSV file we created in Task 1
df = pd.read_csv("match_data.csv")
print("\nLoaded", len(df), "rows from match_data.csv")

# Clean up whitespace in text columns
for col in ["team1", "team2", "venue", "result"]:
    df[col] = df[col].str.strip()

# Remove rows where result is not a team name (unclear results)
all_teams = df["team1"].tolist() + df["team2"].tolist()
df = df[df["result"].isin(all_teams)]
df = df.reset_index(drop=True)

print("Clean rows after filtering:", len(df))


# Step 2 - Create features for the model
# The model needs numbers, not text, so we convert everything

# Target - Did team1 win? (this is what we want to predict)
# 1 = team1 won, 0 = team2 won
df["team1_won"] = (df["result"] == df["team1"]).astype(int)
print("\nTeam1 won:", df["team1_won"].sum(), "| Team2 won:", (df["team1_won"] == 0).sum())


# Feature 1 - Is team1 playing at home?
HOME_VENUES = {
    "sri lanka":   ["colombo", "kandy", "galle", "dambulla", "pallekele"],
    "west indies": ["sabina", "bridgetown", "port of spain", "providence", "kingston"],
    "oman":        ["oman", "muscat", "al amerat"],
    "singapore":   ["singapore"],
    "bahrain":     ["bahrain", "manama"],
    "hong kong":   ["hong kong", "kowloon"],
    "nepal":       ["kathmandu", "kirtipur"],
    "malaysia":    ["kuala lumpur", "malaysia"],
    "rwanda":      ["rwanda", "kigali"],
    "botswana":    ["botswana", "gaborone"],
}

def check_home(row):
    venue = row["venue"].lower()
    team = row["team1"].lower()
    keywords = HOME_VENUES.get(team, [team.split()[0]])
    for keyword in keywords:
        if keyword in venue:
            return 1
    return 0

df["is_home"] = df.apply(check_home, axis=1)


# Feature 2 - How often does each team win in our dataset?
win_counts = df["result"].value_counts().to_dict()
total_matches = len(df)

df["team1_win_rate"] = df["team1"].apply(lambda t: win_counts.get(t, 0) / total_matches)
df["team2_win_rate"] = df["team2"].apply(lambda t: win_counts.get(t, 0) / total_matches)


# Feature 3 - Normalize top score
max_score = df["top_score"].replace("N/A", 0).astype(float).max()
df["top_score_norm"] = df["top_score"].replace("N/A", 0).astype(float) / max_score


# Feature 4 - Convert team names and venues to numbers
team_encoder = LabelEncoder()
team_encoder.fit(list(df["team1"]) + list(df["team2"]))

venue_encoder = LabelEncoder()
venue_encoder.fit(df["venue"])

df["team1_num"] = team_encoder.transform(df["team1"])
df["team2_num"] = team_encoder.transform(df["team2"])
df["venue_num"] = venue_encoder.transform(df["venue"])


# Final feature list and target
features = ["is_home", "team1_win_rate", "team2_win_rate", "top_score_norm", "team1_num", "team2_num", "venue_num"]

X = df[features]
y = df["team1_won"]

print("\nFeatures used:", features)


# Step 3 - Define all the algorithms we want to try
algorithms = {
    "Logistic Regression":    LogisticRegression(max_iter=1000),
    "Decision Tree":          DecisionTreeClassifier(random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight="balanced"),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
}


# Step 4 - Generic function to evaluate any algorithm using cross validation
# I used 5-Fold Stratified Cross Validation
# This splits the data into 5 parts - trains on 4 parts, tests on 1 part
# It repeats this 5 times so every row gets tested exactly once
# StratifiedKFold makes sure both classes (win/loss) appear in every fold

def evaluate_model(name, model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # cross_val_predict runs the model on each fold and returns all predictions
    all_pred = cross_val_predict(model, X, y, cv=cv)
    all_true = y.tolist()

    accuracy = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)

    return accuracy, f1, all_true, list(all_pred)


# Step 5 - Run all algorithms and collect their scores
print("\n" + "=" * 50)
print("Comparing all algorithms using 5-Fold Cross Validation...")
print("=" * 50)

results = {}

for name, model in algorithms.items():
    accuracy, f1, all_true, all_pred = evaluate_model(name, model, X, y)
    results[name] = {
        "accuracy": accuracy,
        "f1": f1,
        "true": all_true,
        "pred": all_pred
    }
    print(name, "-> Accuracy:", round(accuracy * 100, 1), "% | F1 Score:", round(f1, 2))


# Step 6 - Pick the best algorithm based on accuracy
best_name = max(results, key=lambda name: results[name]["accuracy"])
best = results[best_name]

print("\n" + "=" * 50)
print("Best Algorithm:", best_name)
print("Accuracy :", round(best["accuracy"] * 100, 1), "%")
print("F1 Score :", round(best["f1"], 2))
print("=" * 50)


# Step 7 - Print full results for the best algorithm
print("\nConfusion Matrix:")
cm = confusion_matrix(best["true"], best["pred"])
print(cm)

print("\nFull Classification Report:")
print(classification_report(best["true"], best["pred"], target_names=["Team2 won", "Team1 won"], zero_division=0))


# Step 8 - Show feature importance only if best model supports it
best_model = algorithms[best_name]
best_model.fit(X, y)

if hasattr(best_model, "feature_importances_"):
    print("Feature Importance (what the model found useful):")
    for feature, importance in zip(features, best_model.feature_importances_):
        print(" ", feature, "->", round(importance, 3))

print("\nDone!")