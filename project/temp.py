import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Example DataFrame (replace with your actual df_final)
df = pd.read_csv('temp.csv')
df_final=df.copy()
df_final = df_final.dropna(subset='radius_val')
st.title("Interactive Plot: radius_val vs a^(1/n) with Linear Fit")

# Slider for n-th root
n = st.slider("Select n for the n-th root of 'a'", min_value=1, max_value=10, value=2, step=1)

# Compute the n-th root
df_final['a_n_root'] = df_final['a'] ** (1 / n)

# Prepare data for linear regression
X = df_final[['radius_val']].values  # feature
y = df_final['a_n_root'].values      # target

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Compute R^2
r2 = r2_score(y, y_pred)

# Create plot
fig, ax = plt.subplots()
ax.scatter(df_final['radius_val'], df_final['a_n_root'], alpha=0.7, label='Data')
ax.plot(df_final['radius_val'], y_pred, color='red', label='Fit')
ax.grid(ls='--',alpha=0.5)

# Equation text
slope = model.coef_[0]
intercept = model.intercept_
eq_text = f"y = {slope:.3f} x + {intercept:.3f}\n$R^2$ = {r2:.3f}"
ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=12, verticalalignment='top')

ax.set_xlabel('radius_val')
ax.set_ylabel(f"a^(1/{n})")
ax.set_title(f"radius_val vs a^(1/{n}) with Linear Fit")
ax.legend()

st.pyplot(fig)
