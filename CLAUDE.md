# chest-xray-pneumonia-cnn

CNN for detecting pneumonia from chest X-ray images.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Run tests: `pytest tests/ -v`

Train: `python cnn/train.py`

GradCAM viz: `python cnn/gradcam.py`

- Model defined in `cnn/model.py`
- No class-level docstrings. Casual commits.
