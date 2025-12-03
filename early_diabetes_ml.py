import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder


def log_reg(x_train, x_test, y_train, y_test):
    classif = LogisticRegression()
    classif.fit(x_train, y_train)
    y_pred = classif.predict(x_test)
    print("=========log regres==========")
    print(accuracy_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    return y_pred


def KNN(x_train, x_test, y_train, y_test):
    classif = KNeighborsClassifier(n_neighbors=6)
    classif.fit(x_train, y_train)
    y_pred = classif.predict(x_test)
    print("=========KNN==========")
    print(accuracy_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    return y_pred


def Dec_tree(x_train, x_test, y_train, y_test):
    classif = DecisionTreeClassifier(
        min_samples_leaf=6,
        min_samples_split=6,
    )
    classif.fit(x_train, y_train)
    y_pred = classif.predict(x_test)
    print("=========DT==========")
    print(accuracy_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    return y_pred


def RF(x_train, x_test, y_train, y_test):
    classif = RandomForestClassifier(
        min_samples_leaf=6,
        min_samples_split=6,
    )
    classif.fit(x_train, y_train)
    y_pred = classif.predict(x_test)
    print("=========RF==========")
    print(accuracy_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    return y_pred


if __name__ == "__main__":
    df = pd.read_csv("diabetes_data.csv", sep=";")
    df["gender"] = df["gender"].replace(["Male", "Female"], [0, 1])

    df.info()
    describe = df.describe()
    describe.to_csv("describe.csv")

    fig1, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    sns.countplot(ax=axes[0], x="class", data=df)
    axes[0].title.set_text("График 1. Наличие диабета")

    sns.countplot(ax=axes[1], x="class", data=df, hue="gender")
    axes[1].title.set_text("График 2. Пол")

    sns.countplot(ax=axes[2], x="class", data=df, hue="age")
    axes[2].title.set_text("График 3. Возраст")

    fig2, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    sns.countplot(ax=axes[0], x="class", data=df, hue="alopecia")
    axes[0].title.set_text("График 4. Облысение")

    sns.countplot(ax=axes[1], x="class", data=df, hue="obesity")
    axes[1].title.set_text("График 5. Ожирение")

    sns.countplot(ax=axes[2], x="class", data=df, hue="weakness")
    axes[2].title.set_text("График 6. Постоянная слабость")

    fig3, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    sns.countplot(ax=axes[0], x="class", data=df, hue="polyuria")
    axes[0].title.set_text("График 7. Полюрия")

    sns.countplot(ax=axes[1], x="class", data=df, hue="polydipsia")
    axes[1].title.set_text("График 8. Полидипсия")

    sns.countplot(
        ax=axes[2],
        x="class",
        data=df,
        hue="sudden_weight_loss",
    )
    axes[2].title.set_text("График 9. Неожиданная потеря веса")

    fig4, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    sns.countplot(ax=axes[0], x="class", data=df, hue="polyphagia")
    axes[0].title.set_text("График 10. Полифагия")

    sns.countplot(ax=axes[1], x="class", data=df, hue="visual_blurring")
    axes[1].title.set_text("График 11. Помутнение зрения")

    sns.countplot(ax=axes[2], x="class", data=df, hue="itching")
    axes[2].title.set_text("График 12. Зуд")

    fig5, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    sns.countplot(ax=axes[0], x="class", data=df, hue="delayed_healing")
    axes[0].title.set_text("График 13. Медленная регенерация")

    sns.countplot(ax=axes[1], x="class", data=df, hue="partial_paresis")
    axes[1].title.set_text("График 14. Частичный парез")

    sns.countplot(ax=axes[2], x="class", data=df, hue="muscle_stiffness")
    axes[2].title.set_text("График 15. Ригидность мышц")

    fig6, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    axes.title.set_text("График 16. Корреляция")
    sns.heatmap(df.corr().round(2), annot=True, cbar=False)

    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(
        df[df.columns[:-1]],
        df[df.columns[-1]],
        train_size=0.8,
        random_state=0,
    )

    y1 = log_reg(x_train, x_test, y_train, y_test)
    y2 = KNN(x_train, x_test, y_train, y_test)
    y3 = Dec_tree(x_train, x_test, y_train, y_test)
    y4 = RF(x_train, x_test, y_train, y_test)
