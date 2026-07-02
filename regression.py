import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
from sklearn import datasets

df = load_diabetes(as_frame=True).frame
X = df[["bmi"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Simple Linear Regression")
print("Intercept:", round(model.intercept_, 3))
print("bmi coefficient:", round(model.coef_[0], 3))
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("MSE:", round(mean_squared_error(y_test, pred), 3))
print("R^2:", round(r2_score(y_test, pred), 3))

# Plot
plt.figure(figsize=(6, 4))
plt.scatter(X_test["bmi"], y_test, color="blue", alpha=0.6)
plt.plot(sorted(X_test["bmi"]), model.predict(X_test.sort_values("bmi")), color="red")
plt.xlabel("BMI")
plt.ylabel("Target")
plt.title("Simple Linear Regression")
plt.tight_layout()
plt.savefig("simple_linear_regression.png")
plt.close()

# Multiple linear regression
features = ["bmi", "age", "bp"]
X2 = df[features]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)

print("\nMultiple Linear Regression")
print("Intercept:", round(model2.intercept_, 3))
for name, coef in zip(features, model2.coef_):
    print(f"{name} coefficient:", round(coef, 3))
print("MAE:", round(mean_absolute_error(y2_test, pred2), 3))
print("MSE:", round(mean_squared_error(y2_test, pred2), 3))
print("R^2:", round(r2_score(y2_test, pred2), 3))

plt.figure(figsize=(6, 4))
plt.bar(features, model2.coef_, color="purple")
plt.axhline(0, color="black", linewidth=1)
plt.ylabel("Coefficient")
plt.title("Multiple Linear Regression Coefficients")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("multiple_linear_regression.png")
plt.close()

print("\nDone")
