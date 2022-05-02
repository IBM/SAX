import argparse
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from BPR import BusinessProcessRules
from ExtendedLIME import ExtendedLime

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='../data_sample.csv', help='Path to csv data file')
    parser.add_argument('--label', type=str, default='done_accept', help='Target feature name')
    args = parser.parse_args()

    nan_filler = -1

    # Read the data :
    pdf = pd.read_csv(args.filename)
    # Replace NaNs with nan_filler :
    pdf.fillna(nan_filler, inplace=True)
    # Separate features from label :
    X = pdf.drop(columns=[args.label])
    Y = pdf[[args.label]]

    # Split into training and test sets :
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=3, shuffle=True, stratify=Y)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    Y_train.reset_index(drop=True, inplace=True)
    Y_test.reset_index(drop=True, inplace=True)

    # Train a simple (decision tree) model :
    clf = DecisionTreeClassifier(random_state=3).fit(X_train.values, Y_train.values)

    # Get feature Names
    feature_names = X_train.columns

    # Create BP rules object- this object must contain the check_validity and fix_sample methods
    bpr = BusinessProcessRules(feature_names, nan_filler)

    # Initialize the explainer :
    el = ExtendedLime(X_train.values,
                      feature_names=X_train.columns,
                      class_names=['Reject', 'Approve'],
                      categorical_names={3: ['False(novice)', 'True'],
                                         4: ['False(risk)', 'True']},
                      categorical_features=[3, 4],
                      discretize_continuous=False,
                      sample_around_instance=True,
                      random_state=3,
                      nan_filler=nan_filler,
                      bpr=bpr)

    res = dict()
    for ind in range(5):
        res[ind] = el.explain_instance(X_test.iloc[ind].values, clf.predict_proba, bpr_edit=True, bpr_prune=True)
        print(f'result for test instance #{ind}:\n{res[ind].as_list()}')
    return


# Example call:
# python Example.py --filename "data_sample.csv" --label "done_accept"

if __name__ == "__main__":
    main()
