###############################################################
###  A Human Lying Posture Pressure-Map Dataset (LPPDat)  ####
###############################################################
DIRECTORI:
git_root
|-Data
| |-pressure_maps
| | |-0001.png
| | |-0002.png
| | |-.....png
| | |-1279.png
| | |-1280.png
| |-Measuremetns.csv
| |-Targets.csv
|-Documentation
| |-subClass_images
| | |-left_lateral
| | | |-12_01.jpg
| | | |-...
| | | |-12_16.jpg
| | |-prone
| | | |-14_01.jpg
| | | |-...
| | | |-14_16.jpg
| | |-right_lateral
| | | |-13_01.jpg
| | | |-...
| | | |-13_16.jpg
| | |-supine
| | | |-11_01.jpg
| | | |-...
| | | |-11_16.jpg
| |-Information_note.docx
|-Scripts
| |-assess_classifiers_bed_data_occlusions.m
| |-newBedData20.mat
|-README.txt

###############################################################
INFORMATION

{pressure_maps} folder contains 1280 (.png) pressure images. Lables for pictures are integrated in the "Targets.csv" table

{Measurement.csv} is bonus information about the personal dimensions of all participants
[SubjectId, Sex, Age, Height, Weight, Hip, Chest, Waist, Shoulders, Note]
|...SubjectId:	Specifie the subject person metadata
|...Sex, Age, Height, Weight, Hip, Chest, Waist: Shoulders more specifically "Information.docx"

{Targets.csv} contains labels for classification and pressure image identification.
[FileName, Class, SubClass, SubjectId]
|...FileName: 	Identification of pressure image file in "pressure_maps" folder
|...Class:	Main laying posture label: 
|					|...1 = supine
|					|...2 = left lateral
|					|...3 = right lateral
|					|...4 = prone
|...SubClass:	Identification of variation of main postures 1-16 specified in section "Documentation/subClass_images"
|...SubjectId:	Segregates a dataset of pressure map images by study participants for LOSO verification.
###############################################################
AFFILIATION

All authors are employees of Brno University of Technology (Czech Republic), Faculty of Electrical Engineering and Communications, Department of Control and Instrumentation.

###############################################################
ETHICS DECLARATION

Measurements were performed on humans after prior approval of the measurement process by the ethics committee.

###############################################################
LICENSE

The dataset is licensed by CC BY-NC-SA 4.0