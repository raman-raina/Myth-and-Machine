from data_augmentation import augment
from model import train_model, predict_images

d1 = r"C:\Users\raina\Desktop\image_classification_project\data\dataset" 
d2 = r"C:\Users\raina\Desktop\image_classification_project\data\new" 

def main():
    #zip(d1, d2)
    #augment(d1)
    model, classes = train_model(d1)
    predictions = predict_images(d2, model, classes)
    for img, label in predictions.items():
        print(f'Prediction for {img}: {label}')

main()