# A guide to Multi-gpu and Distributed TensorFlow with Estimators API (better name?)

> **NOTE:** This guide is intended for *advanced* users of TensorFlow
and assumes expertise and experience in machine learning.

## Introduction

In this guide we'll go through a full code implementation of a ResNet using
the Estimators API to classify images from CIFAR-10, which is a popular
dataset for image classification. This model is ready to run on a CPU,
multiple GPUs, and also multiple hosts.

The focus is not the model itself, the biggest contribution of this guide is
a practical example of how to build a distributed and multi-gpu model with
TensorFlow high-level APIs, and what to expect to see as results when doing so.

We assume you're already familiar with:
  * [Basic Estimators](https://www.tensorflow.org/extend/estimators)
  * [Distributed Tensorflow concepts](https://www.tensorflow.org/deploy/distributed)

Also, check to this tutorial section before reading this guide:
  * [Training a model using multiple gpu cards](https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards)

## Dataset and Model Overview

CIFAR-10 classification is a common benchmark problem in machine learning.  The
problem is to classify RGB 32x32 pixel images across 10 categories:
```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
```

For more details refer to the [CIFAR-10 page](http://www.cs.toronto.edu/~kriz/cifar.html)
and a [Tech Report](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
by Alex Krizhevsky.

The reason CIFAR-10 was selected was that it is complex enough to exercise
much of TensorFlow's ability to scale to large models. At the same time,
the model is small enough to train fast, which is ideal for trying out
new ideas and experimenting with new techniques.

The model will be a ResNet as proposed in:
```
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385
```

```
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_1/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_2/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_3/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_4/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_5/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_6/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/avg_pool/: (?, 16, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_2/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_3/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_4/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_2/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_3/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_4/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_5/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_6/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1/avg_pool/: (?, 32, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_1/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_2/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_3/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_4/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_5/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_6/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/global_avg_pool/: (?, 64)
INFO:tensorflow:image after unit resnet/tower_0/fully_connected/: (?, 11)
```

### Goals

1. Highlights a canonical organization for network architecture, training and
   evaluation using the Estimators API.
2. Provides a template for constructing larger and more sophisticated models.

### Highlights of the Tutorial

* Complete code implementation that runs on local CPU, GPUs and on multiple hosts;
* Explanation about how to run distributed TensorFlow using Experiments;
* Shows how to create your own Hook;
* Practical example of a input function built with the Dataset API;
* Shows how to generate TFRecord files;

We hope that this guide provides a launch point for building general models
with the Estimators API implementing multi-gpu and distributed support.

## Code Organization

The code for this tutorial resides in
[`models/tutorials/image/cifar10_estimator/`](https://www.tensorflow.org/code/tensorflow_models/tutorials/image/cifar10_estimator/).

File | Purpose
--- | ---
[`generate_cifar10_tfrecords.py`](https://www.tensorflow.org/code/tensorflow_models/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py) | Generates TFRecords from the Python CIFAR-10 data.
[`cifar10.py`](https://www.tensorflow.org/code/tensorflow_models/tutorials/image/cifar10_estimator/cifar10.py) | Input function implementation, reads the TFRecords generated by `generate_cifar10_tfrecords.py`.
[`cifar10_model.py`](https://www.tensorflow.org/code/tensorflow_models/tutorials/image/cifar10_estimator/cifar10_model.py) | Builds the ResNet model.
[`cifar10_main.py`](https://www.tensorflow.org/code/tensorflow_models/tutorials/image/cifar10_estimator/cifar10_main.py) | Trains a CIFAR-10 model on a CPU, GPU, multiple GPUS and even in multiple machines.

## TFRecord

TensorFlow supports different types of file formats as input: CSV files, Fixed
length records and Standard TensorFlow format (TFRecords). You can see more
about it
[here](https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files).

The recommended format for TensorFlow is a [TFRecords](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details)
file. A TFRecords file represents a sequence of (binary) strings. The format
is not random access, so it is suitable for streaming large amounts of data
but not suitable if fast sharding or other non-sequential access is desired.

To create TFRecord files you need to create a
`[tf.python_io.TFRecordWriter](https://www.tensorflow.org/api_docs/python/tf/python_io/TFRecordWriter)`
for each file you want to write:

```python
record_writer = tf.python_io.TFRecordWriter(output_path)
```

You write a series of `tf.Example` protos to the file. Each example contain
a dictionary of features. Each feature can be a `FloatList`, `Int64List` or
`ByteList`.

So, for example, you might encode a single numpy-array image as:

```python
image_str = image.tostring()
image_str = tf.train.BytesList(value=[image_str])
image_str = tf.train.Feature(bytes_list=image_str)

width = tf.train.Int64List(value=[image.shape[0]])
width = tf.train.Feature(int64_list=width)

height = tf.train.Int64List(value=[image.shape[1]])
height = tf.train.Feature(int64_list=height)

features = {
    'height': height,
    'width': width,
    'image_str': image_str}
features = tf.train.Features(feature=features)

example = tf.train.Example(features=features)
writer.write(example.SerializeToString())
```

Full examples for popular datasets are available:
  * CIFAR-10: `[generate_cifar10_tfrecords.py](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py)`
  * MNIST: `[tensorflow/examples/how_tos/reading_data/fully_connected_reader.py](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py)`

## Running Distributed TensorFlow with Estimators

Estimators are a high Level abstraction that support all the basic
operations you need on a Machine Learning model, they also implement
best practices, are distributed by design and are ready to be deployed.

In this guide we'll focus on the distributed feature of Estimators.

In other to run the model on a distributed way using
data-parallelism you can just create an experiment. Experiments know how to
invoke train and eval in a sensible fashion for distributed training.

```python
def get_experiment(estimator, train_input, eval_input):
  def _experiment_fn(run_config, hparams):
    """Creates experiment.

    Experiments perform training on several workers in parallel,
    in other words Experiments know how to invoke train and eval
    in a sensible fashion for distributed training.

    We first prepare an estimator, and bundle it
    together with input functions for training and evaluation
    then collect all that in an Experiment object.
    """
    del run_config, hparams  #unused args
    return tf.contrib.learn.Experiment(
        estimator,
        train_input_fn=train_input,
        eval_input_fn=eval_input
    )
  return _experiment_fn

# run training and evaluation using an Experiment
learn_runner.run(get_experiment(estimator, train_input, eval_input),
                 run_config=run_config)
```

### Set TF_CONFIG

Considering that you already have multiple hosts configured, all you need is a `TF_CONFIG`
environment variable on each host. You can set up the hosts manually or check [tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) for instructions about how to set up a Cluster.

The `TF_CONFIG` will be used by the `RunConfig` to know the existing hosts and their task: `master`, `ps` or `worker`.

Here's an example of `TF_CONFIG`.

```python
cluster = {'master': ['master-ip:8000'],
           'ps': ['ps-ip:8000'],
           'worker': ['worker-ip:8000']}

TF_CONFIG = json.dumps(
  {'cluster': cluster,
   'task': {'type': master, 'index': 0},
   'model_dir': 'gs://<bucket_path>/<dir_path>',
   'environment': 'cloud'
  })
```

*Cluster*

A cluster spec, which is basically a dictionary that describes all of the tasks in the cluster. More about it [here](https://www.tensorflow.org/deploy/distributed).

In this cluster spec we are defining a cluster with 1 master, 1 ps and 1 worker.

* `ps`: saves the parameters among all workers. All workers can read/write/update the parameters for model via ps.
        As some models are extremely large the parameters are shared among the ps (each ps stores a subset).

* `worker`: does the training.

* `master`: basically a special worker, it does training, but also restores and saves checkpoints and do evaluation.

*Task*

The Task defines what is the role of the current node, for this example the node is the master on index 0
on the cluster spec, the task will be different for each node. An example of the `TF_CONFIG` for a worker would be:

```python
cluster = {'master': ['master-ip:8000'],
           'ps': ['ps-ip:8000'],
           'worker': ['worker-ip:8000']}

TF_CONFIG = json.dumps(
  {'cluster': cluster,
   'task': {'type': worker, 'index': 0},
   'model_dir': 'gs://<bucket_path>/<dir_path>',
   'environment': 'cloud'
  })
```

*Model_dir*

This is the path where the master will save the checkpoints, graph and TensorBoard files.
For a multi host environment you may want to use a Distributed File System, Google Storage and DFS are supported.

*Environment*

By the default environment is *local*, for a distributed setting we need to change it to *cloud*.

### Running script

Once you have a `TF_CONFIG` configured properly on each host you're ready to run on distributed settings.

#### Master

## Distributed

* Talk about experiments and how we're running distributed
* https://stackoverflow.com/questions/41600321/distributed-tensorflow-the-difference-between-in-graph-replication-and-between
* https://www.tensorflow.org/deploy/distributed

### Input function

The input function will define our input pipeline implementation, and it's
basically a function that manipulates the data and returns the features and
labels that will be used by the estimator for training, evaluation, and
prediction.

An efficient and scalable way to implement your own input function is to use the
[Dataset API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/datasets.md).

The Dataset API enables you to build complex input pipelines from simple,
reusable pieces, making it easy to deal with large amounts of data, different
data formats, and complicated transformations.

Here's an input function implementation using the Dataset API to read a
TFRecord file.

```python
# Gets TFRecord from disk and repeats dataset undefinitely.
dataset = tf.contrib.data.TFRecordDataset(filenames).repeat()

# Parse records.
dataset = dataset.map(self.parser, num_threads=batch_size,
                      output_buffer_size=2 * batch_size)

# Potentially shuffle records.
if self.subset == 'train':
  min_queue_examples = int(
      Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
  # Ensure that the capacity is sufficiently large to provide good random
  # shuffling.
  dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

# Batch it up.
dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()
image_batch, label_batch = iterator.get_next()

return image_batch, label_batch
```

The Dataset API introduces two new abstractions to TensorFlow: **datasets**
and **iterators**.

* A Dataset can either be a source or a transformation:
  * Creating a source (e.g. Dataset.from_tensor_slices()) constructs a dataset
    from one or more tf.Tensor objects.
  * Applying a transformation constructs a dataset from one or more
    tf.contrib.data.Dataset objects.
    * Repeat: produce multiple epochs;
    * Shuffle: it maintains a fixed-size buffer and chooses the next element
      uniformly at random from that buffer;
    * Batch: constructs a dataset by stacking consecutive elements of another
      dataset into a single element;
    * Map: applies a function to each element in.

* A Iterator provides the main way to extract elements from a dataset.
  The Iterator.get_next() operation yields the next element of a Dataset, and
  typically acts as the interface between input pipeline code and your model.

## Training and Evaluation in a Distributed Environment

Another great thing about Estimators is that they are built to be easily
distributed. 
Below is the code to create and run an experiment.

### Visualizing results with TensorFlow

When using Estimators you can also visualize your data in TensorBoard, with no changes in your code. You can use TensorBoard to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.

You'll see something similar to this if you "point" TensorBoard to the `model_dir` you used to train or evaluate your model.

```shell
# Check TensorBoard during training or after it.
# Just point TensorBoard to the model_dir you chose on the previous step
# by default the model_dir is "sentiment_analysis_output"
$ tensorboard --log_dir="sentiment_analysis_output"
```
## What to expect

* Results table

```shell
# Run this on master:
# Runs an Experiment in sync mode on 4 GPUs using CPU as parameter server for 40000 steps.
# It will run evaluation a couple of times during training.
# The num_workers arugument is used only to update the learning rate correctly.
# Make sure the model_dir is the same as defined on the TF_CONFIG.
$ python cifar10_main.py --data_dir=gs://path/cifar-10-batches-py \
                         --model_dir=gs://path/model_dir/ \
                         --is_cpu_ps=True \
                         --force_gpu_compatible=True \
                         --num_gpus=4 \
                         --train_steps=40000 \
                         --sync=True \
                         --run_experiment=True \
                         --num_workers=2
```
#### Worker

```shell
# Run this on worker:
# Runs an Experiment in sync mode on 4 GPUs using CPU as parameter server for 40000 steps.
# It will run evaluation a couple of times during training.
# Make sure the model_dir is the same as defined on the TF_CONFIG.
$ python cifar10_main.py --data_dir=gs://path/cifar-10-batches-py \
                         --model_dir=gs://path/model_dir/ \
                         --is_cpu_ps=True \
                         --force_gpu_compatible=True \
                         --num_gpus=4 \
                         --train_steps=40000 \
                         --sync=True
                         --run_experiment=True
```

#### PS

```shell
# Run this on ps:
# The ps will not do training so most of the arguments won't affect the execution
$ python cifar10_main.py --run_experiment=True --model_dir=gs://path/model_dir/

# There are more command line flags to play with; check cifar10_main.py for details.
```

## Visualizing results with TensorFlow

When using Estimators you can also visualize your data in TensorBoard, with no changes in your code. You can use TensorBoard to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.

You'll see something similar to this if you "point" TensorBoard to the `model_dir` you used to train or evaluate your model.

```shell
# Check TensorBoard during training or after it.
# Just point TensorBoard to the model_dir you chose on the previous step
# by default the model_dir is "sentiment_analysis_output"
$ tensorboard --log_dir="sentiment_analysis_output"
```

