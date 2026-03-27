import cv2

def preprocessing(image, new_height, new_width):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (new_height, new_height))
    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0
    return normalized