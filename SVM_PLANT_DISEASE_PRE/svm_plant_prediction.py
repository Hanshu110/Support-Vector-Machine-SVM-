import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler

# Function to extract features from an image (Example: color histograms)
def extract_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate color histograms (features)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_sat = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    
    # Flatten and concatenate histograms to form the feature vector
    feature_vector = np.concatenate((hist_hue.flatten(), hist_sat.flatten(), hist_val.flatten()))
    
    return feature_vector

# Function to load dataset (images of healthy and diseased plants)
def load_dataset(image_folder_healthy, image_folder_diseased):
    features = []
    labels = []
    
    # Process healthy plant images
    for img_name in os.listdir(image_folder_healthy):
        img_path = os.path.join(image_folder_healthy, img_name)
        feature = extract_features(img_path)
        if feature is not None:
            features.append(feature)
            labels.append(0)  # 0 for healthy
    
    # Process diseased plant images
    for img_name in os.listdir(image_folder_diseased):
        img_path = os.path.join(image_folder_diseased, img_name)
        feature = extract_features(img_path)
        if feature is not None:
            features.append(feature)
            labels.append(1)  # 1 for diseased
    
    return np.array(features), np.array(labels)

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Function to predict on multiple images in a folder
def predict_multiple_images(image_folder, svm_classifier, scaler):
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        feature = extract_features(img_path)
        
        if feature is None:
            print(f"Error: Could not load or process the image: {img_name}")
            continue
        
        # Standardize the feature
        feature_scaled = scaler.transform([feature])
        
        # Predict using the SVM classifier
        prediction = svm_classifier.predict(feature_scaled)
        
        # Output the prediction
        if prediction == 0:
            print(f"The plant in the image {img_name} is Healthy.")
        else:
            print(f"The plant in the image {img_name} is Diseased.")

# Main function
def main():
    # Load dataset
    healthy_image_folder = 'C:/users/Admin/OneDrive/Desktop/SVM_PLANT_DISEASE_PREDICTION/healthy_plants'
    diseased_image_folder = 'C:/users/Admin/OneDrive/Desktop/SVM_PLANT_DISEASE_PREDICTION/diseased_plants'
    
    features, labels = load_dataset(healthy_image_folder, diseased_image_folder)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train SVM Classifier
    svm_classifier = SVC(kernel='linear', random_state=42, probability=True)
    svm_classifier.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = svm_classifier.predict(X_test)
    
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC Curve
    y_pred_prob = svm_classifier.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
    plot_roc_curve(y_test, y_pred_prob)
    
    # Predict on multiple images in a folder
    new_image_folder = 'C:/users/Admin/OneDrive/Desktop/SVM_PLANT_DISEASE_PREDICTION/input_images'  # Replace with the path to the folder of new images
    predict_multiple_images(new_image_folder, svm_classifier, scaler)

if __name__ == "__main__":
    main()
