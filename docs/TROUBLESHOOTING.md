# Troubleshooting Guide
## Movie Recommendation System Using Restricted Boltzmann Machine (RBM)

---

### Issue 1: PowerShell Script Execution Policy Error
**Symptom:** When running `.\venv\Scripts\Activate.ps1`, PowerShell outputs:
`File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system.`

**Solution:**
Windows restricts unauthorized script execution by default. Run the following command in PowerShell to bypass the policy for the current terminal session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then re-run:
```powershell
.\venv\Scripts\Activate.ps1
```

---

### Issue 2: Port 8501 is Already in Use
**Symptom:** `OSError: [Errno 10048] error while attempting to bind on address ('localhost', 8501)` or Streamlit opens on port 8502.

**Solution:**
Specify an alternative port using the `--server.port` flag:
```powershell
streamlit run app.py --server.port 8505
```

---

### Issue 3: Dataset Download Fails or Times Out (Offline Environment)
**Symptom:** `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>` or network connection error.

**Solution:**
The dataset has already been pre-downloaded and unpacked inside `data/ml-100k/`.
If you ever clone this repository fresh without internet access:
1. Download `ml-100k.zip` from https://files.grouplens.org/datasets/movielens/ml-100k.zip
2. Extract the archive into the `data/` folder so the path `data/ml-100k/u.data` exists.
3. The loader will automatically detect local files without attempting network calls.

---

### Issue 4: `ModuleNotFoundError: No module named 'models'` or `'utils'`
**Symptom:** Python throws `ModuleNotFoundError` when running scripts.

**Solution:**
Always run commands using the `-m` flag from the project root directory:
```powershell
# Correct:
python -m models.train
python -m unittest discover -s tests

# Avoid:
# python models/train.py  (can cause relative import confusion)
```

---

### Issue 5: PyTorch CUDA / GPU Warning
**Symptom:** `UserWarning: CUDA initialization: The NVIDIA driver on your system is too old...`

**Solution:**
This project is specifically designed to run on standard laptop CPUs. The model only has ~170,000 parameters and takes less than 5 seconds to train on CPU. GPU is completely optional and not required.

---

### Issue 6: Matplotlib GUI Warning / Freezing
**Symptom:** Streamlit hangs when rendering charts or throws `UserWarning: Starting a Matplotlib GUI outside of the main thread`.

**Solution:**
In `utils/visualization.py`, we explicitly declare `matplotlib.use("Agg")` before importing `pyplot`. This disables interactive desktop windows and renders pure raster figures in-memory safely.

