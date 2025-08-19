import subprocess
try:
    subprocess.run(["streamlit", "run", "streamlit_app.py"])
except KeyboardInterrupt:
    print("Streamlit app closed by user.")