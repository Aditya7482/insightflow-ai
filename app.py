import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 InsightFlow AI")
st.subheader("Intelligent Sales Analytics & Prediction System")

st.write(
    "A Data Science project for analyzing sales performance, "
    "visualizing business trends, and predicting sales using Machine Learning."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error("data.csv was not found. Please keep data.csv in the same folder as app.py.")
    st.stop()


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("🔎 Filters")

regions = ["All"] + sorted(df["Region"].unique().tolist())
categories = ["All"] + sorted(df["Category"].unique().tolist())

selected_region = st.sidebar.selectbox(
    "Select Region",
    regions
)

selected_category = st.sidebar.selectbox(
    "Select Category",
    categories
)


filtered_df = df.copy()

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == selected_region
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == selected_category
    ]


# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

st.header("📌 Business Overview")

col1, col2, col3, col4 = st.columns(4)

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = len(filtered_df)
average_sales = filtered_df["Sales"].mean()

col1.metric(
    "Total Sales",
    f"₹{total_sales:,.0f}"
)

col2.metric(
    "Total Profit",
    f"₹{total_profit:,.0f}"
)

col3.metric(
    "Orders",
    total_orders
)

col4.metric(
    "Average Sales",
    f"₹{average_sales:,.0f}"
)


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("📋 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("### Dataset Preview")
    st.dataframe(
        filtered_df,
        use_container_width=True
    )

with col2:
    st.write("### Dataset Information")

    st.write(
        f"**Rows:** {filtered_df.shape[0]}"
    )

    st.write(
        f"**Columns:** {filtered_df.shape[1]}"
    )

    st.write(
        f"**Missing Values:** {filtered_df.isnull().sum().sum()}"
    )

    st.write(
        f"**Duplicate Rows:** {filtered_df.duplicated().sum()}"
    )


# --------------------------------------------------
# SALES BY CATEGORY
# --------------------------------------------------

st.header("📈 Sales Analysis")

col1, col2 = st.columns(2)

with col1:

    category_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    category_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Sales by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Sales")

    plt.xticks(rotation=30)

    st.pyplot(fig)


# --------------------------------------------------
# SALES BY REGION
# --------------------------------------------------

with col2:

    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    region_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Sales by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Sales")

    plt.xticks(rotation=30)

    st.pyplot(fig)


# --------------------------------------------------
# PROFIT ANALYSIS
# --------------------------------------------------

st.header("💰 Profit Analysis")

profit_data = (
    filtered_df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots()

profit_data.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Profit by Category")
ax.set_xlabel("Category")
ax.set_ylabel("Profit")

plt.xticks(rotation=30)

st.pyplot(fig)


# --------------------------------------------------
# TOP CUSTOMERS
# --------------------------------------------------

st.header("🏆 Top Customers")

top_customers = (
    filtered_df.groupby("Customer")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

st.dataframe(
    top_customers.reset_index(),
    use_container_width=True
)


# --------------------------------------------------
# MACHINE LEARNING MODEL
# --------------------------------------------------

st.header("🤖 Sales Prediction Model")

st.write(
    "A Random Forest Regression model is used to predict sales "
    "based on quantity, discount, profit, category, and region."
)


model_df = df.copy()

X = model_df[
    [
        "Quantity",
        "Discount",
        "Profit",
        "Category",
        "Region"
    ]
]

y = model_df["Sales"]


# Convert categorical variables to numerical values
X = pd.get_dummies(
    X,
    columns=["Category", "Region"]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(
    X_train,
    y_train
)


# Predictions
predictions = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


col1, col2 = st.columns(2)

col1.metric(
    "Mean Absolute Error",
    f"{mae:.2f}"
)

col2.metric(
    "R² Score",
    f"{r2:.2f}"
)


# --------------------------------------------------
# PREDICTION TABLE
# --------------------------------------------------

prediction_df = pd.DataFrame(
    {
        "Actual Sales": y_test.values,
        "Predicted Sales": np.round(predictions, 2)
    }
)

st.write("### Actual vs Predicted Sales")

st.dataframe(
    prediction_df,
    use_container_width=True
)


# --------------------------------------------------
# AUTOMATED INSIGHTS
# --------------------------------------------------

st.header("💡 Automated Business Insights")

best_category = (
    df.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)

best_region = (
    df.groupby("Region")["Sales"]
    .sum()
    .idxmax()
)

best_customer = (
    df.groupby("Customer")["Sales"]
    .sum()
    .idxmax()
)


st.success(
    f"📦 Highest sales category: **{best_category}**"
)

st.info(
    f"🌍 Highest sales region: **{best_region}**"
)

st.warning(
    f"👤 Highest-value customer by sales: **{best_customer}**"
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "InsightFlow AI | Built with Python, Pandas, Scikit-learn and Streamlit | "
    "B.Tech Data Science Project"
)