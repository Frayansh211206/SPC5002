import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('../data/retail_orders_week1.csv')
df.shape
df = df[df['quantity'] > 0]
df = df[df['delivery_days'] > 0]
df['order_value'] = df['order_value'] * 1.2
len(df)
sample = df.sample(1200,random_state=7)
sample['returned'].mean()
features = ['item_price', 'discount_pct', 'quantity',
            'delivery_days', 'customer_prior_orders']

X = df[features]
y = df['returned']

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=7, stratify=y)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print('ROC-AUC', round(auc, 3))
coefs = pd.Series(model.coef_[0], index=features).sort_values()
coefs
summary = (
    df.groupby('category', as_index=False)['returned']
      .mean()
      .rename(columns={'returned': 'return_rate'})
      .sort_values('return_rate', ascending=False)
)
summary