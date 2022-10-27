from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def predict_from_existing_model(model_name="pre_trained_XGboost_model.json", test_set=[[]]):
    model = XGBClassifier()
    model.load_model(model_name)
    y_pred = model.predict(test_set)
    predictions = [round(value) for value in y_pred]
    return predictions


def train_new_model(data_file = 'data_generated.csv', seed =7, test_percentage =0.2):
    # The last column in the file is the target column
    #  All entries should be numbers
    dataset = loadtxt(data_file, delimiter=",", skiprows=True)
    n = len(dataset)
    m = len(dataset[0])
    print("Num of rows in the dataset ", n)
    print("Num of columns in the dataset ", m)
    X = dataset[:, 0:m - 2]
    Y = dataset[:, m - 1]
    # Define hyper parameters
    print("build train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_percentage, random_state=seed)
    # fit model no training data
    model = XGBClassifier()
    print("Fitting XGB model")
    model.fit(X_train, y_train)
    model.save_model(data_file+".model.json")
    print("Predicting")
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    F1 = f1_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("F1: %.2f%%" % (F1 * 100.0))
    return predictions


if __name__ == "__main__":
    print("Running - Do not forget to uncomment one of the below lines")
    # train_new_model()             # Uncomment if you want to train a new model on your dataset
    # predict_from_existing_model() # Uncomment if you want to use our pre-trained model for your prediction


