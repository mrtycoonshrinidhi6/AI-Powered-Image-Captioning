---

# AI-Powered Image Captioning

Automatically generate natural-language captions for images using a CNN+RNN deep learning pipeline and a small Streamlit-based demo app.

This repository contains the code, notebooks and artifacts used to build, train and run an image-captioning model. The project extracts visual features from images (CNN), then feeds them into a sequence model (LSTM) to produce descriptive captions.

## Highlights

- CNN-based visual feature extraction (pretrained backbone, e.g., VGG16 or similar)
- LSTM-based caption generator (sequence model trained on paired images and captions)
- Jupyter notebooks for preprocessing, model building, and prediction
- A simple Streamlit app (`app.py`) for demoing predictions locally
- A saved model artifact in `Model Building/model.h5` for quick inference

## Repository structure

- `app.py` - Streamlit demo application to upload an image and show generated caption
- `requirements.txt` - Python dependencies for the project
- `images/` - Example images used for testing and demos
- `Model Building/` - Notebooks and artifacts for training. Contains `model.h5` (trained model)
   - `modelBuilding.ipynb` - Notebook used to prepare data and train the model
- `Prediction/` - Notebooks for running inference and visualizing outputs
   - `Prediction and Visualization.ipynb`
- `Preprocessing/` - Notebooks documenting image and text preprocessing steps
   - `Image-preprocessing.ipynb`
   - `Description_preprocessing.ipynb`

> Note: There are additional project folders in the workspace (e.g., other projects). This README focuses on the `AI-Powered-Image-Captioning` folder contents.

## Quickstart (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip; pip install -r requirements.txt
```

3. Run the demo Streamlit app:

```powershell
streamlit run app.py
```

4. Open the local URL printed by Streamlit (usually http://localhost:8501) and upload an image to see generated captions.

## Using the pre-trained model for quick inference

If you want to use the provided model (`Model Building/model.h5`) from a notebook or a script:

- Load the model with Keras/TensorFlow
- Make sure any preprocessing steps (image resizing, normalization, tokenization) match those used during training
- Run the model to generate integer token outputs, then map tokens back to words using the saved tokenizer/vocabulary

See `Prediction/Prediction and Visualization.ipynb` for an example inference workflow.

## Notebooks

- `Preprocessing/*` — shows how images and captions were prepared (resizing, feature extraction, tokenization, vocabulary creation)
- `Model Building/modelBuilding.ipynb` — training loop, model definition, callbacks, and saving artifacts
- `Prediction/Prediction and Visualization.ipynb` — running inference and visualizing captions on images

Open these notebooks in Jupyter or VS Code to reproduce steps or adapt the pipeline.

## Dataset and Training

This repository does not include a large external dataset by default. Typical datasets used for image captioning are MS COCO or Flickr8k/Flickr30k. If you plan to retrain:

- Acquire a dataset of images with multiple reference captions (COCO, Flickr)
- Preprocess images and captions according to the notebooks in `Preprocessing/`
- Update paths and hyperparameters in `Model Building/modelBuilding.ipynb` as needed

Training a model end-to-end can be time-consuming and may require a GPU. For most users, using the provided `model.h5` and running inference in the `Prediction` notebook or `app.py` is sufficient.

## Troubleshooting

- If the Streamlit app fails to start, ensure `streamlit` is installed in the active environment and that you activated the environment before running.
- If model loading fails, verify your TensorFlow/Keras version matches the version used to produce `model.h5` (you may need to re-save the model in a compatible format).
- Tokenizer/vocabulary mismatch will cause incorrect captions — make sure the tokenizer used at inference matches the one used for training.

## Contribution

Contributions are welcome. If you want to improve the project:

1. Fork the repository
2. Create a branch for your feature: `git checkout -b feature/your-feature`
3. Implement and verify changes (update notebooks, tests, README)
4. Open a pull request describing your changes

Please ensure any added code is well-documented and notebooks produce reproducible outputs.

## License & Attribution

This project is provided as-is for educational and experimental purposes. Add a license file (e.g., `LICENSE`) to make reuse terms explicit. If you reuse external datasets or pretrained models, follow their respective licenses and attribution requirements.

## Contact

If you have questions or want to collaborate, open an issue in the repository or contact the maintainer listed in the repo metadata.

---

If you'd like, I can also:

- Add a small README section showing example inputs/outputs (screenshot or sample images and captions)
- Create a requirements subset for the demo (smaller `requirements_demo.txt`)
- Add a tiny script to run inference from the command line (e.g., `predict.py`)

Tell me which of these you'd like next.

