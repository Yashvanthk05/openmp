import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re

st.set_page_config(page_title="Mark Visualizer", layout="wide")

st.title("📈 VIT MARK VISUALISER")

uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv","xlsx"])

if uploaded:
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    elif uploaded.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded)
        csv_path = "converted_file.csv"
        df.to_csv(csv_path, index=False)
        df = pd.read_csv(csv_path)
        st.success(f"✅ Excel file converted to CSV and saved as {csv_path}")
    st.write("### 📝 Preview of Uploaded Data", df)

    target_column = st.selectbox("🎯 Select Target Column (y)", df.columns)
    features = st.multiselect("📐 Select Feature Columns (X)", [col for col in df.columns if col != target_column])
    clean_method = st.radio("🧹 Choose Cleaning Method", ("Fill Missing with Mean", "Drop Missing Rows"))

    if st.button("🚀 Clean & Run Gradient Descent"):
        X = df[features].select_dtypes(include=[np.number])
        y = df[target_column].astype(float)

        if X.empty or y.empty:
            st.error("❌ Selected columns must be numeric.")
        else:
            if clean_method == "Fill Missing with Mean":
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())
            elif clean_method == "Drop Missing Rows":
                combined = pd.concat([X, y], axis=1).dropna()
                X = combined[features]
                y = combined[target_column]

            X = (X - X.mean()) / X.std()
            y = y.values.reshape(-1, 1)

            X.to_csv("X.csv", index=False, header=False)
            pd.DataFrame(y).to_csv("y.csv", index=False, header=False)

            st.write("### Cleaned Data Preview")
            st.write("#### Features (X):")
            st.write(X.head())
            st.write("#### Target (y):")
            st.write(pd.DataFrame(y).head())

            with st.spinner("⚙️ Running Gradient Descent with OpenMP..."):
                os.system("gcc -fopenmp gd_omp.c -o gd_omp -lm")
                result = subprocess.run(
                    ["./gd_omp", str(X.shape[1]), "1000", "0.01"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                st.success("✅ Gradient Descent Completed!")

            st.write("### 📋 Output from OpenMP Program")
            st.code(result.stdout)
            if result.stderr:
                st.error(result.stderr)

            # cost_log = pd.read_csv("cost_log.csv")
            # st.write("### 📉 Cost vs. Epoch")
            # fig, ax = plt.subplots()
            # ax.plot(cost_log['epoch'], cost_log['cost'], label="Cost")
            # ax.set_xlabel("Epoch")
            # ax.set_ylabel("Cost")
            # ax.grid(True)
            # st.pyplot(fig)
            stats = {}
            lines = result.stdout.splitlines()
            for line in lines:
                if "Mean" in line:
                    stats["Mean"] = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                elif "Standard Deviation" in line:
                    stats["Std Dev"] = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                elif "Min" in line:
                    stats["Min"] = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                elif "Max" in line:
                    stats["Max"] = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                elif "Median" in line:
                    stats["Median"] = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())

            # Plot target column
            st.write("### 🎯 Plot of Target Values (y) with Statistical Annotations")

            fig, ax = plt.subplots(figsize=(6, 2))  # Set figure size to 500x900 pixels (in inches: 5x9)
            ax.plot(y, marker='o', linestyle='-', color='blue', label='Target Values')

            # Annotate stats
            for label, value in stats.items():
                ax.axhline(y=value, linestyle='--', label=f'{label}: {value:.2f}')

            ax.set_title("Target Column with Statistics")
            ax.set_xlabel("Index")
            ax.set_ylabel("Target Value")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

st.markdown("---")
st.header("🔓 ZIP Password Cracker (with OpenMP)")

zip_file = st.file_uploader("📦 Upload a password-protected ZIP file", type=["zip"])
pwd_length = st.number_input("🔑 Maximum Password Length", min_value=1, max_value=10, value=4, step=1)

if st.button("🔐 Start Cracking ZIP Password"):
    if zip_file is None:
        st.warning("⚠️ Please upload a .zip file.")
    else:
        zip_path = "protected.zip"
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        st.info("⚙️ Compiling password cracker...")
        compile_status = os.system("gcc -o password_cracker password_cracker.c -lzip -lm -fopenmp")

        if compile_status != 0:
            st.error("❌ Compilation failed. Ensure libzip is installed.")
        else:
            st.success("✅ Compilation Successful!")

            with st.spinner("🔍 Cracking password..."):
                result = subprocess.run(
                    ["./password_cracker", str(pwd_length), zip_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            st.success("✅ Cracking Completed!")
            st.code(result.stdout)
            if result.stderr:
                st.error(result.stderr)
