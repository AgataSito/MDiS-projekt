import functions
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

MINCR = 0


def run():
    colormap_bright = plt.cm.RdBu

    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(3, weights='distance'),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
    }

    results = []

    # data = functions.get_data_from_api(MINCR)
    df = functions.get_data_from_json(MINCR)

    sns.countplot(y='cr', data=df).set(title="Amount of creatures per CR")
    df.info()  # cr pomimo ze float - jest tu kategoria, name nie jest potrzebne
    plt.show()

    sns.boxplot(x='hp', y='cr', data=df).set(title="HP to CR relation")
    plt.show()
    sns.boxplot(x='ac', y='cr', data=df).set(title="AC to CR relation")
    plt.show()

    # musze przekonwertowac pandas na sklearn (poprzez numpy)
    df_cr = df['cr'].copy()
    df.drop(['cr', 'name'], axis=1, inplace=True)
    if df_cr.isnull().values.sum() != 0:
        df_cr = df_cr.fillna(df_cr['cr'].value_counts().index[0])

    x = StandardScaler().fit_transform(df)
    encoder = OneHotEncoder(sparse=False, drop='first')
    sk_cr = encoder.fit_transform(df_cr.values.reshape(-1, 1))
    y = sk_cr

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    y_train_inv = encoder.inverse_transform(y_train).ravel()
    y_test_inv = encoder.inverse_transform(y_test).ravel()
    axis = plt.subplot(1, len(classifiers) + 1, 1)

    axis.set_title("Input data")
    axis.scatter(x_train[:, 0], x_train[:, 1], c=y_train_inv, cmap=colormap_bright, edgecolors="k")
    axis.scatter(x_test[:, 0], x_test[:, 1], c=y_test_inv, cmap=colormap_bright, alpha=0.6, edgecolors="k")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(())
    axis.set_yticks(())

    for classifier_idx, (classifier_name, classifier) in enumerate(classifiers.items()):
        axis = plt.subplot(1, len(classifiers) + 1, 2 + classifier_idx)
        classifier.fit(x_train, y_train)
        y_pred_inv = encoder.inverse_transform(classifier.predict(x_test)).ravel()
        axis.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=y_pred_inv,
            cmap=colormap_bright,
            edgecolors="k",
            alpha=0.6,
        )

        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks(())
        axis.set_yticks(())
        axis.set_title(classifier_name)
        results.append({
            "Classifier": classifier_name,
            "Accuracy": accuracy_score(y_test_inv, y_pred_inv),
            "Precision": precision_score(y_test_inv, y_pred_inv, average="weighted"),
            "Recall": recall_score(y_test_inv, y_pred_inv, average="weighted"),
            "F1": f1_score(y_test_inv, y_pred_inv, average="weighted"),
            "MCC": matthews_corrcoef(y_test_inv, y_pred_inv),
        })

    plt.tight_layout()
    plt.show()
    results_pd = pd.DataFrame(results)
    print(results_pd)


if __name__ == '__main__':
    run()
