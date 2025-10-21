# Task 1 — Simple Linear Regression (Basic Level)

### Description
In this task, a **Simple Linear Regression** model is implemented using **Python** and **scikit-learn** to understand the relationship between two continuous variables.

We use a real-world dataset — the **Advertising Dataset**, which explores how different advertising budgets (TV, Radio, Newspaper) impact product sales. For simplicity, this task focuses only on the relationship between **TV advertising spend** and **Sales**.

### Objective
To build a linear regression model that:
- Learns the relationship between TV advertising spend and product sales.
- Visualizes the regression line.
- Makes predictions for new values of advertising spend.

### Key Steps
1. Import and explore the dataset.
2. Select relevant columns: `TV` (independent variable) and `Sales` (dependent variable).
3. Split the dataset into **training** and **testing** sets.
4. Train a **Linear Regression** model using scikit-learn.
5. Evaluate the model using **Mean Squared Error (MSE)** and **R² Score**.
6. Plot the regression line and actual data points.
7. Predict sales for a new TV advertising budget.

### Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

### Output
- A plotted regression line showing the relationship between TV spend and sales.
- Model evaluation metrics (MSE and R²).
- Predicted sales for new advertising budgets.
