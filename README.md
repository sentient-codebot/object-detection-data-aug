# object-detection-data-aug

hi all. this is for the convenience of doing a Computer Vision project (reproducing yolo v1). 
The module *datatrans* contains transforms that apply on images and bounding boxes together. 

bounding box format: 
[n, 5] tensor, n = # objects, second dimension contains: [category, x (relative to image width), y (relative to image height), width, height]

combined with the customized *dataset* (not written by me), it should work fijn. 
