# Dataset
`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend `Fashion-MNIST` to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Here's an example how the data looks (*each class takes three-rows*):

![](doc/img/fashion-mnist-sprite.png)

# Models & performance

Final average accuracy on test dataset: 94.22% (over 5 repetitions)

This has been obtained by training for 100 epochs and saving the models with lowest validation losses.

I also tried an ensemble of these models, which yielded 94.86% accuracy.

# [Demo](https://drive.google.com/file/d/1_4bvgIwhTv0_P-BhrG4WdfYNN8W-1ZaX/view)
Once I the model was trained, I used it to classify the pieces of clothing real-time using video cam. 
The performance leaves a lot to be desired, which is due to own - fast coded - segmentation method, but nonetheless it was a cool exercise.

Link to [DEMO](https://drive.google.com/file/d/1_4bvgIwhTv0_P-BhrG4WdfYNN8W-1ZaX/view) 

Link to [PDF summary](https://github.com/JakubCzerny/fashion-mnist-assignment/blob/master/assignment.pdf)

Link to [notebook for training](https://github.com/JakubCzerny/fashion-mnist-assignment/blob/master/my_code/training.ipynb)

Run life demo:

`cd` to /fashion-mnist-assignment/my_code 

run `python camera.py`

