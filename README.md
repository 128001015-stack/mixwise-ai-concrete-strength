# MixWise AI — Concrete Strength Intelligence Studio

A student-ready Streamlit web application based on the supplied UCI Concrete Compressive Strength dataset.

## Features
- Custom MixWise AI identity and responsive dashboard
- Random Forest regression with an 80/20 held-out test split (random_state=42)
- Strength prediction from all 8 dataset inputs
- R², MAE and RMSE metrics
- Actual-vs-predicted plot, residual plot and feature importance
- Dataset table, downloadable CSV, histograms, scatter plots and correlation heatmap
- Mix Design Explorer that screens water/superplasticizer combinations for a target strength
- Built-in AI Engineering Assistant for model and concrete mix-design concepts
- Prominent engineering disclaimer

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:
   `pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`

## Deploy online (Streamlit Community Cloud)
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `Concrete_Data.xls`, and this README.
2. In Streamlit Community Cloud, create an app from that repository and choose `app.py` as the entry point.
3. Deploy. The first launch trains the model, so it may take a short while.

## Dataset citation
Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5PK67. Dataset is licensed CC BY 4.0.
Official dataset page: https://archive.ics.uci.edu/dataset/165/concrete%2Bcompressive%2Bstren

## Important engineering disclaimer
Predictions and candidate combinations are educational screening estimates, not certified mix designs. The application does not verify all requirements of IS 10262, IS 456, durability/exposure conditions, aggregate grading, workability, admixture compatibility, or material-specific properties. Validate through a qualified engineer and laboratory trial mixes before implementation.
