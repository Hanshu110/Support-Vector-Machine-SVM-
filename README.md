# Support-Vector-Machine-SVM-
A Support Vector Machine (SVM) is a powerful supervised learning algorithm used for binary classification. In this project, we apply SVM to classify plants as either healthy or diseased based on features extracted from images.

**Implementing a Support Vector Machine (SVM) for Plant Health Classification**

This project involves developing a Support Vector Machine (SVM) classifier to distinguish between healthy and diseased plants based on image features.

#### Prerequisites

Before running the project, ensure the following Python libraries are installed:

- `opencv-python`: For image processing.
- `scikit-learn`: For machine learning algorithms, including SVM.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.

Install them using pip:

```bash
pip install opencv-python scikit-learn numpy matplotlib seaborn
```

#### Running the Project

1. **Prepare the Dataset**:
   - Create three directories:
     - `healthy_plants`: Contains images of healthy plants.
     - `diseased_plants`: Contains images of diseased plants.
     - `input_images`: Contains the image of input plants.
   - Ensure each directory has a sufficient number of images for training and testing.

2. **Update the Script**:
   - Modify the `healthy_image_folder`, `diseased_image_folder` and `input_images` variables in the script to point to your dataset directories.

3. **Execute the Script**:
   - Run the Python script:

     ```bash
     python svm_plant_prediction.py
     ```

   - The script will:
     - Load and preprocess the images.
     - Extract features using color histograms.
     - Split the data into training and testing sets.
     - Train the SVM classifier.
     - Evaluate the model's performance.
     - Display a confusion matrix and ROC curve with health condition of the plant(healthy / diseased).

#### Understanding SVM

Support Vector Machines (SVMs) are supervised learning models used for classification and regression tasks. They work by finding the optimal hyperplane that separates data points of different classes in a high-dimensional space. SVMs are effective in high-dimensional spaces and are widely used for classification tasks due to their ability to model complex relationships citeturn0search13.

In this project, SVMs are utilized to classify plant images into healthy or diseased categories based on extracted image features and predicts the condition of the plant.
