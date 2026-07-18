import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def calibrate_threshold(human_scores, target_fpr=0.05):
    human_scores = np.asarray(human_scores)
    return float(np.quantile(human_scores, 1.0 - target_fpr))


def train_human_vs_bot(human_df, bot_df, feature_cols, random_state=42, name=None):
    dataset = pd.concat(
        [human_df.assign(is_bot=0), bot_df.assign(is_bot=1)],
        ignore_index=True,
    )
    input_data = dataset[feature_cols]
    output_data = dataset["is_bot"]

    input_train, input_test, output_train, output_test = train_test_split(
        input_data, output_data, test_size=0.2, random_state=random_state, stratify=output_data
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(input_train, output_train)
    output_pred = model.predict(input_test)
    acc = accuracy_score(output_test, output_pred)

    tag = f" === {name} ===" if name else ""
    print(f"In-domain test accuracy{tag}: {acc:.2%}  (baseline 50%)")
    print(classification_report(output_test, output_pred, target_names=["human", "bot"]))
    return model, acc
