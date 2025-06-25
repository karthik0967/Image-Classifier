
# Image Classifier

This project implements image classification using **Support Vector Machines (SVM)** to classify images into two categories: **cars** 🚗 and **jet fighters** ✈️. The dataset is pre-organized into folders for each class, and images are resized, flattened, and fed into a machine learning model for training and testing.

---

## 📁 Project Structure

```
Image-Classifier/
├── Image Data/
│   ├── car/
│   └── jet fighter/
├── main.ipynb
├── README.md
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (SVM, GridSearchCV)
- Scikit-image (`skimage`)

---

## 🔍 Workflow Overview

1. **Data Preparation**  
   - Load image data from directories.
   - Resize all images to a consistent shape `(150x150x3)`.
   - Flatten images for SVM input.

2. **Label Encoding**  
   - `0` for `car`  
   - `1` for `jet fighter`

3. **Model Training**  
   - Split dataset into train and test sets.
   - Use SVM with GridSearchCV to tune hyperparameters.

4. **Evaluation**  
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

5. **Prediction on New Images**  
   - Accepts a URL for a new image.
   - Predicts the class and displays the resized image.

---

## 🧪 Sample Results

- **Accuracy**: 94.44%
- **Confusion Matrix**:
  ```
  [[10  0]
   [ 1  7]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

             0       0.91      1.00      0.95        10
             1       1.00      0.88      0.93         8

      accuracy                           0.94        18
     macro avg       0.95      0.94      0.94        18
  ```

---

### 🔁 Option 1: Using in Google Colab

1. **Upload your dataset to Google Drive**, e.g. to `/MyDrive/ML/Datasets/Image Data`
2. **Mount Google Drive in Colab** with:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Set the path**:
   ```python
   DATADIR = '/content/drive/MyDrive/ML/Datasets/Image Data'
   ```
4. Run the notebook cells to train and test the model.

---

### 💻 Option 2: Running Locally

1. **Place the dataset folder (`Image Data/`) in the same directory** as `main.ipynb`
2. **Set the path**:
   ```python
   DATADIR = './Image Data'
   ```
3. Make sure you have all dependencies installed:
   ```bash
   pip install numpy pandas matplotlib scikit-learn scikit-image
   ```
4. Run the notebook using Jupyter or any Python IDE.

---

## 🖼️ Example Input & Output

**Input Image URL**:  
`https://image.cnbcfm.com/api/v1/image/101669400-12795614153_dfc68d6c52_o.jpg?v=1500062421`

**Prediction**:  
```
Predicted output: jet fighter
```

---

## 📌 Notes

- Resize and flattening may reduce image quality, but it simplifies input for classical ML models.
- Consider using CNNs for better performance in future extensions.
- You may expand the classifier by adding more categories and images.

---

## 👤 Author

**Karthik**  

---
