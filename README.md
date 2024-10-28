# Deep-Learning-alpjai-Main-Project

**Team name:**  BHAF

**Names and Neptun codes of the team members:**  
- Bilszky Márk, C0QVQN  
- Sági Máté, PGBWYA  
- Pesti Patrik, E329CD

**Project name:**  Airbus Ship Detection Challenge

### **Project description:**  
This project focuses on detecting ships in satellite images and drawing aligned bounding boxes around them. The dataset includes a mix of images, some containing no ships, while others may feature multiple ships of varying sizes. Ships can appear in diverse environments such as open sea, docks, marinas, and more. To accomplish this task, we employ a 2D convolutional neural network (CNN) for accurate detection and localization.

## Files:
### **deepl_bp.ipynb:**
To use this code, you will need to generate a Kaggle API key, which can be downloaded from Kaggle as `kaggle.json`. The code will then download the files associated with the current challenge. Afterward, it creates output images from the RLE data present in the `train_v2.csv` file. The code visualizes the distribution of the number of ships in each image and displays some sample images.

To visualize the input and output together, the code generates a mask from the RLE-encoded pixels and overlays it onto the original image. At the end, the code creates an array containing filenames with a specified proportion of images with and without ships. These filenames serve as tokens to access the corresponding input and output data.

**Note:** This method ensures that we avoid using unnecessary numbers of images without any area of interest. However, the images are selected randomly and can be shuffled at any time.


