import pandas as pd
from sklearn.linear_model import LogisticRegression

employee_data = pd.read_csv("https://tinyurl.com/y6r7qjrp")

# grab independent variables columns
inputs = employee_data.iloc[:, :-1]

#grab dependent "did_quit" variable column
output = employee_data.iloc[:, -1]

# build logistic regression
fit = LogisticRegression(penalty='none').fit(inputs, output)

# Print coefficients:
print("COEFFICIENTS: {0}".format(fit.coef_.flatten()))
print("INTERCEPT: {0}".format(fit.intercept_.flatten()))

# Interact and test with new employee data
def predict_employee_will_stay(sex, age, promotions, years_employed):
    prediction = fit.predict([[sex, age, promotions, years_employed]])
    probabilities = fit.predict_proba([[sex, age, promotions, years_employed]])
    if prediction[0] == 1:
        return f"WILL LEAVE: {probabilities[0][1]:.2f}"
    else: 
        return f"WILL STAY: {probabilities[0][0]:.2f}"

#Test a prediction
while True: 
    n = input("Predict employee will stay or leave {sex},{age}, {promotions}, {years_employed}: ")
    (sex, age, promotions, years_employed) = n.split(",")
    print(predict_employee_will_stay(int(sex), int(age), int(promotions), int(years_employed)))
# Test the function with new employee data
predict_employee_will_stay(1, 35, 1, 5)
