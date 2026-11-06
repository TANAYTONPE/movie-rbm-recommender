# Complete Installation and Execution Guide
## Movie Recommendation System Using Restricted Boltzmann Machine (RBM)
### Windows Setup Instructions

This guide provides step-by-step instructions for running the project on a Windows laptop. No dedicated GPU is required; all computations are optimized for standard CPU execution.

---

## System Requirements

- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Python Version:** Python 3.10, 3.11, 3.12, 3.13, or 3.14
- **RAM:** Minimum 4 GB (8 GB recommended)
- **Disk Space:** ~500 MB free disk space

---

## Step 1: Open Terminal in Project Folder

1. Open **Windows PowerShell** or **Command Prompt**.
2. Navigate to the project root directory:

```powershell
cd C:\Users\tanay\.gemini\antigravity\scratch\movie-rbm-recommender
```

*(Or right-click the project folder in Windows File Explorer and select **"Open in Terminal"**).*

---

## Step 2: Create a Virtual Environment (Recommended)

Creating a dedicated virtual environment isolates project dependencies:

```powershell
python -m venv venv
```

---

## Step 3: Activate the Virtual Environment

Activate the environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

*Note for PowerShell users:* If you encounter an execution policy error (`running scripts is disabled`), run this command first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then rerun `.\venv\Scripts\Activate.ps1`.

If using standard Command Prompt (`cmd.exe`):
```cmd
venv\Scripts\activate.bat
```

---

## Step 4: Install Required Dependencies

Install the verified dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

This installs:
- `torch` (PyTorch CPU edition)
- `streamlit` (Web dashboard framework)
- `pandas` & `numpy` (Matrix manipulations)
- `scikit-learn` (Metric evaluations)
- `matplotlib` & `pillow` (Visualization and diagram generation)
- `requests` (Data fetching)

---

## Step 5: Verify Automated Unit Tests

Run the test suite to ensure the RBM mathematics, Gibbs sampling, and data loading pass all validation checks:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

**Expected output:**
```
Ran 10 tests in 2.9s
OK
```

---

## Step 6: Train the RBM Model

Execute the model training script. It trains the RBM on the MovieLens 100K dataset using Contrastive Divergence (CD-1) and logs real evaluation metrics:

```powershell
python -m models.train
```

**What this does:**
1. Loads 100,000 ratings across 943 users and 1,682 movies.
2. Trains for 25 epochs using mini-batches ($B=64$, $\text{LR}=0.05$, $\text{Momentum}=0.8$).
3. Computes real test-set RMSE and MAE against held-out ratings.
4. Evaluates Top-K recommendation ranking metrics (Precision, Recall, Hit Rate, MRR).
5. Saves model weights to `saved_models/rbm_movielens100k.pt`.
6. Saves training curves to `saved_models/training_history.json`.

*(Training typically completes in under 10 seconds on a modern CPU).*

---

## Step 7: Launch the Streamlit Web Application

Start the interactive dashboard:

```powershell
streamlit run app.py
```

Streamlit will start a local web server and display:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your default browser should open automatically. If not, open your browser and navigate to `http://localhost:8501`.

---

## Step 8: How to Stop the Application

To shut down the web server:
- Click the PowerShell window where Streamlit is running.
- Press `Ctrl + C`.

---

## Quick Command Summary

```powershell
# 1. Navigate to directory
cd C:\Users\tanay\.gemini\antigravity\scratch\movie-rbm-recommender

# 2. Run automated tests
python -m unittest discover -s tests -p "test_*.py"

# 3. Train RBM model
python -m models.train

# 4. Start web application
streamlit run app.py
```

