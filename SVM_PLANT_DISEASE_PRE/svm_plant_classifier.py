import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def extract_features(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_sat = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    
    feature_vector = np.concatenate((hist_hue.flatten(), hist_sat.flatten(), hist_val.flatten()))
    
    return feature_vector

def load_dataset(image_folder_healthy, image_folder_diseased):
    features = []
    labels = []
    
    for img_name in os.listdir(image_folder_healthy):
        img_path = os.path.join(image_folder_healthy, img_name)
        feature = extract_features(img_path)
        if feature is not None:
            features.append(feature)
            labels.append(0)
    
    for img_name in os.listdir(image_folder_diseased):
        img_path = os.path.join(image_folder_diseased, img_name)
        feature = extract_features(img_path)
        if feature is not None:
            features.append(feature)
            labels.append(1)
    
    return np.array(features), np.array(labels)

def main():
    
    healthy_image_folder = 'C:/users/Admin/OneDrive/Desktop/SVM_PLANT_DISEASE_PREDICTION/healthy_plants'
    diseased_image_folder = 'C:/users/Admin/OneDrive/Desktop/SVM_PLANT_DISEASE_PREDICTION/diseased_plants'
    
    features, labels = load_dataset(healthy_image_folder, diseased_image_folder)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)
    
    y_pred = svm_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
