# SMS Spam Detection

This repository contains code and data for a simple SMS spam detection project. The model and notebook implement preprocessing, feature extraction (CountVectorizer / TfidfVectorizer), and an sklearn classifier to distinguish "spam" vs "ham" messages.

## Repository structure

- `sms_detector.ipynb` — Jupyter notebook with data loading, preprocessing (regex, tokenization, stemming), vectorization, model training, hyperparameter search, and a small GUI demo using `tkinter`.
- `finalized_model.sav` — Trained and serialized model file (pickle format).
- `spam.csv` — Original dataset of SMS messages and labels.
- `requirements.txt` — Python package requirements for running the notebook and reproducing results.

## Dependencies

Install the dependencies from `requirements.txt`. On Windows, using PowerShell:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `tkinter` is part of the standard library on most Windows Python installs (used in the notebook for a small GUI). If you don't have it, install the official python.org distribution or enable the Tkinter option.
- `pickle` is part of the standard library; `joblib` is included for convenience when working with model files.

## NLTK data

The notebook uses NLTK utilities (tokenization, stopwords, PorterStemmer). After installing packages, download required NLTK corpora:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Quick usage

Open the notebook to run the analysis and training steps:

```powershell
jupyter notebook sms_detector.ipynb
```

Or use the serialized model directly from Python to predict a single SMS message:

```python
import pickle
import re

# load model
with open('finalized_model.sav','rb') as f:
    model = pickle.load(f)

# sample preprocessing function (match the notebook's pipeline)
def preprocess_text(text):
    # very small example; adapt to match the notebook exactly
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", '', text)
    return text

msg = "Free entry in 2 a weekly competition to win FA Cup final tickets"  # example
processed = preprocess_text(msg)
# depending on how the model was saved, you may need to vectorize input first
# For example, if the notebook saved a pipeline (vectorizer + classifier) it will accept raw strings
pred = model.predict([processed])
print('Prediction:', pred[0])
```

If the saved `finalized_model.sav` contains a scikit-learn Pipeline including the vectorizer and classifier, you can pass raw text directly. If not, load the vectorizer that was used to transform training data (some projects store it alongside the trained classifier).

## Reproducing training

1. Open `sms_detector.ipynb`.
2. Run the preprocessing and vectorization cells.
3. Run model training and evaluation cells. The notebook uses `GridSearchCV` and `train_test_split` from scikit-learn.

## Troubleshooting

- If a module import fails, ensure you're running the code in the virtual environment that has packages from `requirements.txt` installed.
- If NLTK tokenizers or stopwords are missing, run the `nltk.download(...)` commands shown above.
- On Windows the `tkinter` GUI may require the standard Python installer; the Windows Store distribution of Python sometimes omits it.

## License & Next steps

This project is intended for educational/demonstration purposes. Next steps you could take:
- Add a small CLI script (`predict.py`) that wraps the model for easy use.
- Add unit tests for the preprocessing pipeline.
- Replace pickle with `joblib` for larger models or store as a scikit-learn `Pipeline` that includes vectorizer + classifier.

If you'd like, I can add a minimal `predict.py` that demonstrates loading the model and predicting from the command line.
