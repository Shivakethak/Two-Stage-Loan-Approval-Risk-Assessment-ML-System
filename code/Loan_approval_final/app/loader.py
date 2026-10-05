from pathlib import Path
import joblib


def load_models(config):
    # Project root = Loan_approval_final/
    BASE_DIR = Path(__file__).resolve().parent.parent

    cls_path = BASE_DIR / config['models']['classifier']
    reg_path = BASE_DIR / config['models']['regressor']

    if not cls_path.exists():
        raise FileNotFoundError(f"Classifier not found: {cls_path}")

    if not reg_path.exists():
        raise FileNotFoundError(f"Regressor not found: {reg_path}")

    cls = joblib.load(cls_path)
    reg = joblib.load(reg_path)

    return cls, reg
