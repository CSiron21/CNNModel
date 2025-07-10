# MobileNetV2 Grayscale CNN Model

## Project Structure
```
train_data/
    images/           # Grayscale images for training
    keypoints/        # (Optional) Keypoints data if needed
    labels.csv        # CSV with columns: image,label
cnn_model.py          # Main script for training and exporting
requirements.txt      # Python dependencies
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Organize your data as shown above. `labels.csv` should have at least two columns: `image` (filename in images/) and `label` (class/exercise type).

## Training
Run the following command to train the model:
```bash
python cnn_model.py
```

- Images are resized to 128x128 and normalized to [0,1].
- Model uses MobileNetV2 (customized for grayscale, small alpha).

## Exporting for Mobile
After training, the script automatically converts the model to TensorFlow Lite (`.tflite`) and appends the unique labels to the file for edge deployment.

## Notes
- Adjust `alpha` in the script for a smaller/larger model.
- For custom keypoints or additional outputs, modify the model and data generator accordingly. 