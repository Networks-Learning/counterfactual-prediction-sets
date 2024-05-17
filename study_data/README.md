# **ImageNet16H-PS Dataset**

```rewards_strict.csv``` and ```rewards_lenient.csv``` contain the data that we collected in our human subject study for each image and prediction set under the strict and the lenient implementation of our systems. More specifically, ```rewards_strict.csv``` contains the following columns:

* **image_name**: the name of the image for which the participant was asked to predict the true label, as it appears in the [ImageNet16H dataset](https://osf.io/2ntrf/). Type: ```str```.
* **set**: the size of the prediction set, constructed using conformal prediction for the model VGG-19, from which set the participant had to predict a label value for the given image. Note that given a fixed image and a fixed classifier, there is an one-to-one mapping from any prediction set constructed using conformal prediction to its size, as there can be only one prediction set for each set size. As a result, given the image and the set size one can uniquely identify the prediction set. Type: ```int```. 
* **reward**: the variable indicating whether the participant managed to predict the ground truth label successfully or not. Type: ```bool```.
* **worker_id**: a randomized anonymous identifier of the participant that made the prediction. Type: ```str```.
* **timestamp**: the time that the prediction was submitted. Type: ```str```.

Similarly, ```rewards_lenient.csv``` is organized as follows:
* **image_name**: same as above.
* **set**: the size of the prediction set constructed using conformal prediction that was recommended to the participant in order to predict a label value for the given image. The size does not include the option "Other", which was also shown to the participant. Type: ```int```. 
* **reward**: same as above.
* **is_other**: the variable indicating whether the participant predicted a label value from outside the recommended prediction set. Type: ```bool```.
* **worker_id**: same as above.
* **timestamp**: same as above.

The data are under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
