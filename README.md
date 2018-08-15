# Pedestrian Detection in Images and Videos

### Read Project Report.pdf for more

Matlab GUI & implementation, training and testing of CNNs to detect pedestrians . Under 1s detection, 80% accuracy. (1 Sliding Window VGG) (2 HoG SVM VGG) (3 SSD)

CNN = Convolutional Neural Networks
SSD = Single Shot (Multibox) Detector

General purpose pre-trained models (http://www.vlfeat.org/matconvnet/pretrained/) were trained (fine-tuned) for pedestrian detection.

The performance of each network is evaluated on the same set of 40 test images 640x480 gathered from various labelled machine-learning collections available online. Detection of pedestrians in those images have various level of difficulty due to different levels of occlusion, scale etc.

1.	Sliding window at various scales submitted to CNN and evaluated with softmax using matconvnet-vgg-s
80% detection rate, averaging 22s per picture and 30% false positive rate  

2.	Sliding window HoG-SVM as region proposal. Regions submitted to CNN matconvnet-vgg-s for evaluation
90% detection rate, averaging 1s with 10% false positives

3.	SSD using ssd- pascal-vggvd-300.
98% detection accuracy with an averaged time of 0.7s per image and 5% false positive rate

4. Pedestrian detector in videos using binary image transformations as a region proposal for the CNN matconvnet-vgg-s (1.)
