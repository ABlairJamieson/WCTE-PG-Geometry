
## Geometry to file code 

Use Dean's Geometry package (https://github.com/WCTE/Geometry).  

Geometry_blair.py is a helper/wrapper that runs Dean's geometry package to extract design locations of LEDs, Domes and cameras in WCTE.  

WCTE_Geometry_to_file.py uses Geometry_blair.py to get the geometry, and puts it into a json file.

This github also includes a sample output json file.

Potential future updates:
- output non-design positions
- save both design and non-design positions in the json file

## Predictor Labeller code

Code to label an image using the design LED locations as a guide.  Predicts what an image from the ideal design geometry should look like, then matches LEDs to the closest blob.

PredictImage.py : module of functions to help with labelling the image

Predict_Image_PCH1.ipynb : uses PredictImage.py to do image labelling for an image from PCH1
