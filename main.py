import io
import os
import cv2
import numpy as np
from google.cloud import vision
from google.cloud.vision_v1 import types
from matplotlib import pyplot as plt
from PIL import Image

#Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "segmentation-424411-7bf90199e5dd.json"

#Initialize the Vision API client
client = vision.ImageAnnotatorClient()

def extract_text(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    return texts

def segment_visual_elements(image_path, texts, output_dir):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Use a binary threshold to create a binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    #Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Draw contours on the original image
    segmented_image = image.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)

    #Draw text bounding boxes on the segmented image
    for text in texts[1:]:  #Skip the first item (full text block)
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        vertices = np.array(vertices, dtype=np.int32)
        cv2.polylines(segmented_image, [vertices], True, (255, 0, 0), 2)
        cv2.putText(segmented_image, text.description, (vertices[0][0], vertices[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    #Save segmented visual elements as images
    visual_elements = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        element_path = os.path.join(output_dir, f'element_{i}.png')
        cv2.imwrite(element_path, roi)
        visual_elements.append(element_path)

    #Save the segmented image
    segmented_image_path = os.path.join(output_dir, 'segmented_image.png')
    cv2.imwrite(segmented_image_path, segmented_image)

    return visual_elements, segmented_image_path



#Displays the image with extracted text overlaid
def display_image_with_text(image_path, texts):
    image = cv2.imread(image_path)
    for text in texts[1:]:  #First text is the entire text block
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        cv2.polylines(image, [np.array(vertices, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.putText(image, text.description, vertices[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def generate_html(texts, visual_elements, segmented_image_path, output_html_path):
    html_content = '<html>\n<head>\n<title>Extracted Content</title>\n</head>\n<body>\n'

    #Add extracted text as paragraphs
    for text in texts[1:]:  #Skip the first item (full text block)
        html_content += f'<p>{text.description}</p>\n'

    #Add the segmented image
    html_content += f'<h2>Segmented Image with Text</h2>\n<img src="{segmented_image_path}" alt="Segmented Image">\n'

    #Add visual elements as images
    html_content += '<h2>Visual Elements</h2>\n'
    for element in visual_elements:
        html_content += f'<img src="{element}" alt="Visual Element">\n'

    html_content += '</body>\n</html>'

    #Write to the HTML file
    with open(output_html_path, 'w') as f:
        f.write(html_content)


def main(image_path, output_dir, output_html_path):
    #Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    texts = extract_text(image_path)
    print("Extracted Text:")
    for text in texts:
        print(f'\n"{text.description}"')

    visual_elements, segmented_image_path = segment_visual_elements(image_path, texts, output_dir)
    generate_html(texts, visual_elements, segmented_image_path, output_html_path)


if __name__ == "__main__":
    image_path = 'sample_image.jpg'
    output_dir = 'output_elements'
    output_html_path = 'output.html'
    main(image_path, output_dir, output_html_path)
