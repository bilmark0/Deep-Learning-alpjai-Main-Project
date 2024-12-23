# Deep-Learning-alpjai-Main-Project  

*Team name:* BHAF  

*Names and Neptun codes of the team members:*  
- Bilszky Márk, C0QVQN  
- Sági Máté, PGBWYA  
- Pesti Patrik, E329CD  

*Project name:* Airbus Ship Detection Challenge  

---

## *Project Description*  
This project focuses on detecting ships in satellite images and generating segmentation masks that highlight their exact locations. The dataset includes images with diverse scenarios, such as no ships, or multiple ships of varying sizes.  

To achieve this, we implement a *simplified Mask R-CNN* model with a *ResNet50 backbone* for feature extraction. The network generates binary masks that outline the ships in the input images.  

---

## *Files*  

### **deepl_bp.ipynb**  
This Jupyter Notebook is the core of the project.  

#### Key Features:  
1. *Data Preparation*  
   - *Kaggle Integration*: Downloads the dataset files from Kaggle using an API key (kaggle.json).  
   - *RLE Decoding*: Converts Run-Length Encoded (RLE) masks from the train_v2.csv file into binary masks.  
   - *Image Selection*: Balances the dataset by controlling the proportion of images with and without ships.  

2. *Visualization*  
   - Displays the distribution of ships across the dataset.  
   - Provides visual overlays of the masks on the original satellite images.  

3. *Neural Network Integration*  
   - Implements a *simplified Mask R-CNN* model to generate binary masks for ship detection.  
   - Demonstrates the performance of the model by overlaying predictions on sample input images.  

---

## *Neural Network*  

### *Architecture*  
We use a *simplified Mask R-CNN*, specifically tailored for binary segmentation tasks. The main components of this architecture are:  

1. *Backbone*:  
   - *ResNet50*: A pre-trained convolutional neural network used to extract high-level features from the images.  
   - Outputs are taken from specific layers to provide multi-scale feature maps.  

2. *Mask Head*:  
   - Processes the extracted features using convolution and transpose convolution layers to upscale and refine the mask prediction.  
   - Outputs a binary mask for ship segmentation.  

### *Why This Approach?*  
- *Simplification*: By focusing solely on mask generation, the model avoids the complexity of bounding box regression and classification.  
- *Efficiency*: The ResNet50 backbone leverages transfer learning for robust feature extraction.  
- *Scalability*: The architecture can be extended in the future to include bounding box prediction if needed.  

### *Performance*  
The model currently generates masks that highlight ships, but further tuning is required to improve segmentation accuracy and handle edge cases.  

---

## *Planned Improvements*  
1. *Higher Resolution Output*:  
   - Enhance the mask output to align more precisely with ship boundaries.  
2. *Model Expansion*:  
   - Add bounding box detection and classification for a complete Mask R-CNN implementation.  
3. *Augmentation Techniques*:  
   - Apply data augmentation to make the model more robust to variations in ship appearance and environment.  

---

## *Getting Started*  

### *Requirements*  
- Python 3.x  
- TensorFlow/Keras  
- OpenCV  
- Kaggle API Key (kaggle.json)  

### *Installation*  
1. Clone the repository:  
   
git clone https://github.com/your-repo-url
cd your-repo-name
2. Install dependencies:  
   
pip install -r requirements.txt

### *Run the Notebook*  
- Open the deepl_bp.ipynb file in Jupyter Notebook or any IDE that supports Jupyter.  
- Ensure the Kaggle API key (kaggle.json) is placed in the root directory.  
- Execute the cells step-by-step to train and test the model.  

---
## *Results*  
Here are some sample results from the simplified Mask R-CNN model:  
- *Input Image*: Satellite images from the dataset.  
- *Predicted Output*: Binary masks highlighting ships in the images.  

---

## *Evaluation*  

As for evaluation, plotting the learning curve is challenging due to the long training time per epoch. However, this can be observed directly by running the code or by checking the GitHub repository, where the notebook is available.  

### Key Points:  
1. *Large Backbone Network*:  
   - The ResNet50 backbone includes a significant number of parameters, most of which are not trainable.  
   - This reduces the computational burden while leveraging a pre-trained feature extractor that is already highly optimized for this type of task.  

2. *Advantages of Transfer Learning*:  
   - By using a pre-trained backbone, we avoid the need to train a feature extraction network from scratch.  
   - This approach ensures better performance, even with limited training data, as the backbone is already adept at identifying key features.  

3. *Current Results*:  
   - The ships are detected in the input images, but the masks are currently rough and require refinement.  
   - The focus moving forward will be on improving mask precision and ship boundary alignment, rather than building a feature extraction network from the ground up.  

These points highlight the strengths of the model while identifying areas for further improvement. The backbone ensures feature discovery is handled efficiently, allowing us to direct our efforts toward refining segmentation accuracy.
