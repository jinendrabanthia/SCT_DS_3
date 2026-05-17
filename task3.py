import sys
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def main():
    # ── 1. Load Dataset ──────────────────────────────────────────────
    try:
        df = pd.read_csv('bank.csv', sep=';')
        print("[OK] Dataset loaded successfully.")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    except FileNotFoundError:
        print("[ERROR] bank.csv not found in the current directory.")
        return

    # ── 2. Data Preprocessing ────────────────────────────────────────
    print("[STEP] Preprocessing data...")

    # Separate features and target
    X = df.drop(columns=['y'])
    y = df['y']

    # One-hot encode all categorical variables
    X_encoded = pd.get_dummies(X)

    # Map target to binary (yes=1, no=0)
    y_encoded = y.map({'yes': 1, 'no': 0})

    print(f"   Features after encoding: {X_encoded.shape[1]}")
    print(f"   Target distribution: {dict(y.value_counts())}\n")

    # ── 3. Train / Test Split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded,
        test_size=0.3,
        random_state=42
    )
    print(f"[DATA] Train set: {X_train.shape[0]} samples")
    print(f"[DATA] Test  set: {X_test.shape[0]} samples\n")

    # ── 4. Model Training ────────────────────────────────────────────
    print("[TRAIN] Training Decision Tree Classifier (max_depth=3)...")
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    print("   Model trained successfully.\n")

    # ── 5. Evaluation ────────────────────────────────────────────────
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])

    print("=" * 55)
    print("MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"   Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)\n")
    print("   Classification Report:")
    print("-" * 55)
    print(report)
    print("=" * 55)

    # Top feature importances
    importances = pd.Series(clf.feature_importances_, index=X_encoded.columns)
    top_features = importances.nlargest(5)
    print("\nTOP 5 FEATURE IMPORTANCES:")
    print("-" * 40)
    for feat, imp in top_features.items():
        bar = "#" * int(imp * 40)
        print(f"   {feat:<25s} {imp:.4f}  {bar}")
    print("-" * 40 + "\n")

    # ── 6. Decision Tree Visualization ───────────────────────────────
    print("[VIZ] Generating decision tree visualization...")

    # Premium dark theme
    matplotlib.rcParams.update({
        'figure.facecolor': '#0D1117',
        'axes.facecolor': '#0D1117',
        'text.color': '#E6EDF3',
        'axes.labelcolor': '#8B949E',
        'xtick.color': '#8B949E',
        'ytick.color': '#8B949E',
    })

    fig, ax = plt.subplots(figsize=(28, 12))

    # Draw the tree
    plot_tree(
        clf,
        feature_names=X_encoded.columns.tolist(),
        class_names=['No', 'Yes'],
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True,
        impurity=True,
        ax=ax
    )

    # Title
    fig.suptitle(
        'Decision Tree Classifier — Bank Term Deposit Prediction',
        fontsize=24, fontweight='black', color='#FFFFFF', y=0.97
    )

    # Subtitle
    fig.text(
        0.5, 0.93,
        f'max_depth = 3  |  Accuracy = {accuracy:.2%}  |  UCI Bank Marketing Dataset  |  {df.shape[0]} samples',
        ha='center', fontsize=13, color='#8B949E', style='italic'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_file = 'decision_tree_model.png'
    fig.savefig(output_file, dpi=250, bbox_inches='tight', facecolor='#0D1117', edgecolor='none')
    print(f"[OK] Decision tree saved as '{output_file}'\n")

    plt.close(fig)


if __name__ == "__main__":
    main()
