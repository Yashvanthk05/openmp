import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import re
import time

st.set_page_config(page_title="File Unlocker", layout="wide")

st.header("🔓 File Password Cracker (with OpenMP)")

zip_file = st.file_uploader("📦 Upload a password-protected ZIP file", type=["zip"])
pwd_length = st.number_input("🔑 Maximum Password Length", min_value=1, max_value=10, value=4, step=1)
num_threads = st.slider("🧵 Number of Threads", min_value=1, max_value=os.cpu_count(), value=4)

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

            st.write(f"🔍 Cracking password using {num_threads} threads...")
            with st.spinner(f"Cracking in progress..."):
                result = subprocess.run(
                    ["./password_cracker", str(pwd_length), zip_path,str(num_threads)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            st.success("✅ Cracking Completed!")
            st.code(result.stdout)
            if result.stderr:
                st.error(result.stderr)

if st.button("📊 Benchmark Speedup (1 to Max Threads)"):
    if zip_file is None:
        st.warning("⚠️ Please upload a .zip file.")
    else:
        zip_path = "protected.zip"
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        max_threads = os.cpu_count()
        thread_counts = list(range(1, max_threads + 1))
        execution_times = []

        st.info("⏱️ Benchmarking... This may take some time ⌛")

        progress = st.progress(0)
        for i, threads in enumerate(thread_counts):
            result = subprocess.run(
                ["./password_cracker", str(pwd_length), zip_path, str(threads)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            match = re.search(r"Time taken: ([0-9.]+)", result.stdout)
            if match:
                exec_time = float(match.group(1))
                execution_times.append(exec_time)
            else:
                execution_times.append(None)

            progress.progress((i + 1) / len(thread_counts))

        valid_data = [(t, e) for t, e in zip(thread_counts, execution_times) if e is not None]
        thread_counts, execution_times = zip(*valid_data)
        base_time = execution_times[0]
        speedup = [base_time / t for t in execution_times]

        data = {
            "Number of Threads": thread_counts,
            "Execution Time (s)": execution_times,
            "Speedup": speedup
        }
        df = pd.DataFrame(data)
        
        st.write("### 📋 Benchmark Results")
        st.dataframe(df)
        
        st.write("### ⏱️ Execution Time vs Threads")
        fig1, ax1 = plt.subplots()
        ax1.plot(thread_counts, execution_times, marker='o')
        ax1.set_xlabel("Number of Threads")
        ax1.set_ylabel("Execution Time (s)")
        ax1.set_title("Execution Time vs Threads")
        ax1.grid(True)
        st.pyplot(fig1)
        st.write("### ⚡ Speedup vs Threads")
        fig2, ax2 = plt.subplots()
        ax2.plot(thread_counts, speedup, marker='s', color='green')
        ax2.set_xlabel("Number of Threads")
        ax2.set_ylabel("Speedup")
        ax2.set_title("Speedup vs Threads")
        ax2.grid(True)
        st.pyplot(fig2)

