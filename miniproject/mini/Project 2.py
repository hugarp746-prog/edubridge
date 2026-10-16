import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
data = {
    'CIBIL_Score': [550, 600, 620, 650, 680, 700, 720, 750, 780, 800],
    'Loan_Amount': [500000, 400000, 600000, 300000, 500000,
                    400000, 600000, 500000, 700000, 600000],
    'Eligible': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['CIBIL_Score', 'Loan_Amount']]
y = df['Eligible']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
cibil = 720
amount = 500000

prediction = model.predict([[cibil, amount]])

if prediction[0] == 1:
    print("Person is eligible for the loan.")
else:
    print("Person is not eligible for the loan.")
plt.figure(figsize=(8, 5))

plt.scatter(
    df['CIBIL_Score'],
    df['Loan_Amount'],
    c=df['Eligible'],
    cmap='coolwarm',
    s=100
)

plt.xlabel('CIBIL Score')
plt.ylabel('Loan Amount Required (₹)')
plt.title('Loan Eligibility Prediction Using Logistic Regression')
plt.colorbar(label='Eligibility (0 = No, 1 = Yes)')
plt.grid(True)
plt.show()

