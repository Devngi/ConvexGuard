# ConvexGuard

Fraud detection with a from-scratch SVM. The browser runs the model formula — no server, no API.

## Structure
```text
data/         put creditcard.csv here
train/        train_and_export.py + requirements.txt
site/         index.html + style.css + app.js + data.js + lib/
src/          NumPy SVM, KKT, duality gap, metrics
tests/        3 core tests (solver vs sklearn, KKT, duality)
```

## Quickstart
```bash
# 1. Install dependencies
pip install -r train/requirements.txt
# 2. Train model and export site/data.js
python train/train_and_export.py
# 3. Open site/index.html in browser
open site/index.html
```
