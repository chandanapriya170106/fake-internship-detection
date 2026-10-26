# Machine Learning-Based Fake Internship Detection (+ Blockchain Trust)

## Project structure
```
app.py                       Streamlit web app (ML detection + blockchain tabs)
requirements.txt
data/                        dataset (sample is synthetic; put real CSV here)
src/preprocess.py            cleaning + red-flag features
src/generate_sample_data.py  builds the synthetic demo dataset
src/train.py                 trains LR / RF / SVM / NB, saves best model
src/predict.py               predict one posting
blockchain/ledger.py         simple Python blockchain (SHA-256, proof of work)
contracts/InternshipRegistry.sol   optional Solidity contract (Remix / Sepolia)
models/                      saved model, results.csv, confusion_matrix.png
```

## Run
```bash
pip install -r requirements.txt
python -m src.generate_sample_data     # skip if you have the real dataset
python -m src.train
python -m src.predict                  # quick test
python -m blockchain.ledger            # blockchain demo
streamlit run app.py
```

## IMPORTANT: use a real dataset for your report
The bundled data is SYNTHETIC, so scores of 100% are meaningless. Download the
Kaggle "Real / Fake Job Postings" (EMSCAD) dataset, save it as
`data/fake_job_postings.csv`, and run `python -m src.train` again. The script
uses it automatically. Expect lower, realistic scores (fake class is only ~5%).
Report precision, recall and F1 for the fake class, not just accuracy.

## Pipeline (matches the slides)
Collect data -> Preprocess -> Extract features (TF-IDF + red flags) -> Train -> Predict
