import pandas as pd
import numpy as np
import scipy.stats as st

df = pd.read_csv("House_Price.csv")

target = pd.DataFrame(df["price"])
df.drop("price", axis=1, inplace=True)
df.drop("bus_ter", axis=1, inplace=True)

waterbody_encoded = pd.get_dummies(df["waterbody"], "waterboody")
df.drop("waterbody", axis=1, inplace=True)

df = pd.concat([df, waterbody_encoded], axis=1)
df["airport"] = df["airport"].map({"YES": 1, "NO": 0})

df = df.astype(np.float64)

df["n_hos_beds"] = df["n_hos_beds"].fillna(df["n_hos_beds"].mean())

'''for name in df.columns:
    print(name, np.any(np.isnan(df[name])))'''

correlation_matrix = df.corr()

high_correlation = correlation_matrix[np.abs(correlation_matrix) > 0.9]
upper_triangle = high_correlation.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

if len(to_drop) > 0:
    df = df.drop(columns=to_drop)


def prepXY(X, Y):
    X = np.asarray(X)
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))

    Y = np.asarray(Y)
    #Y = np.expand_dims(Y, 0).T

    return X, Y

def fit(inpX, inpY):
    X = inpX
    y = inpY
    beta_ = np.linalg.inv(X.T @ X) @ X.T @ y
    beta_ = np.expand_dims(beta_, 0).T
    eps_ = (y - X @ beta_)[0]

    sigma_ = np.sum(eps_ ** 2) / (X.shape[0] - X.shape[1])

    return beta_[0], eps_, sigma_

def pred(inpX, beta, eps):
    return inpX @ beta + eps


deleted_columns = []
myX, myY = prepXY(df, target)
while True:
    print("=================================================")
    b, e, s = fit(myX, myY)
    print("Оценка дисперсии ошибок:", s)

    var_beta = np.diag(s * np.linalg.inv(myX.T @ myX))
    print("Дисперсии коэффициентов:", var_beta)

    std_X = np.std(myX[:,1:], axis=0, ddof=1)
    std_Y = np.std(myY, ddof=1)

    beta_hat = b[1:].T[0]
    std_beta_hat = np.sqrt(var_beta[1:])

    standardized_betas = beta_hat * (std_X / std_Y)
    print("Стандартизованные коэффициенты:", standardized_betas)

    t_statistics = beta_hat / std_beta_hat
    df_ = myX.shape[0] - myX.shape[1]
    p_values = 2 * st.t.cdf(-np.abs(t_statistics), df_)

    print("t-статистика:", t_statistics)
    print("p-значения:", p_values)
    if not np.all(p_values < 0.05):
        idx = np.argmax(p_values)
        print(f"Удаляем {df.columns[idx]}")
        deleted_columns.append(df.columns[idx])
        df = df.drop(df.columns[idx], axis=1)
        myX, myY = prepXY(df, target)
    else:
        break


print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("-----------------------------------------------------")
print("Deleted:", deleted_columns)
df = pd.concat([target, df], axis=1)
print(df)
df.to_csv('New_House_Price.csv', index=False)
