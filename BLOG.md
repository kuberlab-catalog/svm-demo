# Remote calls to Tensorflow serving directly from your Golang code.

One of the problem in machine learning is make your model servable, it means make inference process for model. Tensorflow has built-in Serving Server that allows your run serving server automatically, your only need to export your Tensorflow model to appropriate format [TODO: Reference to tensoflow serving documenation](https://kuberlab.com). And you can use [gRPC](https://kuberlab.com) protocol to invoke your  model methods.

So according documentation it looks very simple, like 1,2,3. But devil in details. Actually there are small number examples on python how to make remote calls, no documentation in single place about types and interfaces, you need deep dive to protobuffs files and tensorflow model export code to just make simple things to work. In that blog post I am going to provide basic example how use it from golang.


## Build Model.

Lets start with very simple binary classification problem.  Lets say we have points in 2-Dimension space,(X,Y). And each point belongs to one of two classes RED if X>Y or GREEN if X<=Y.

### Generate training and validation sets.

```
import numpy as np
import pandas as pd
from numpy import random
#Generate training set
train = pd.DataFrame(random.randint(low=0, high=100, size=(1000,2)),columns=['x', 'y'])
train['label'] = train.apply(lambda v: 0 if v['x']>v['y']+(5-random.random_sample()*10) else 1,axis=1)
#Generate validation set
test = pd.DataFrame(random.randint(low=0, high=100, size=(100,2)),columns=['x', 'y'])
test['label'] = test.apply(lambda v: 0 if v['x']>v['y'] else 1,axis=1)
#Save datas to file
dataset_dir = './data'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

train.to_csv(dataset+'/train.csv')
test.to_csv(dataset+'/test.csv')
```

### Build SVM model

```
import tensorflow as tf
from tensorflow import logging
dataset_dir = './data'
train_set = tf.data.TextLineDataset(dataset_dir+'/train.csv')
validation_set = tf.data.TextLineDataset(dataset_dir+'/test.csv')
checkpoint_dir = './training'
with tf.Session():
        def _parse(line):
            row = tf.string_split([line],delimiter=',').values
            i = row[0]
            x = tf.string_to_number(row[1], tf.float32)
            y = tf.string_to_number(row[2], tf.float32)
            label = tf.string_to_number(row[3], tf.int32)
            return {'i':tf.reshape(i,[1]),'x':tf.reshape(x,[1]),'y':tf.reshape(y,[1])},tf.reshape(label,[1])

        train_set = train_set.skip(1).map(_parse).shuffle(100).repeat()
        validation_set = validation_set.skip(1).map(_parse)
        def _input_fn():
            return train_set.make_one_shot_iterator().get_next()
        def _validation_fn():
            return validation_set.make_one_shot_iterator().get_next()
        featureX = tf.contrib.layers.real_valued_column('x')
        featureY = tf.contrib.layers.real_valued_column('y')
        model = tf.contrib.learn.SVM(example_id_column='i',
                                         feature_columns=[featureX, featureY],
                                         model_dir=checkpoint_dir,
                                         l1_regularization=l1_regularization,
                                         l2_regularization=l2_regularization)
        model.fit(input_fn=_input_fn, steps=steps)
        metrics = model.evaluate(input_fn=_validation_fn)
        logging.info("Done:\nValidation Loss: %.4f\nValidation Accuracy: %.4f",metrics['loss'],metrics['accuracy'])
    feature_spec = {'x': tf.FixedLenFeature(dtype=tf.float32,shape=[1]),
                'y': tf.FixedLenFeature(dtype=tf.float32,shape=[1])}
    #serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    #there is bug in tensoflow export_savedmodel implementaion.It doesn't undertand tf.estimator.export.build_parsing_serving_input_receiver_fn return type, So we are using depricated method just beacase ot works.
    serving_fn = tf.contrib.learn.utils.input_fn_utils.build_parsing_serving_input_fn(feature_spec)
    export_path = model.export_savedmodel(export_dir_base=checkpoint_dir,serving_input_fn=serving_fn)
    logging.info("\nModel Path: %s",export_path.decode("utf-8") )
```


Ok now our model has been built and exported to directory './training/{model_version}

### Start model sering

As I already mention above it is pretty simple:
```
tensorflow_model_server --port=9000 --model_name=my_svm_mode --model_base_path=./training
```
You only need installed tensorflow_model_server, or you can use [one of our pre-built](https://hub) docker container for it.
```
docker run --rm -it -p 9000:9000 -v {ABSOLUTE_PATH_TO_YOUR_TRAINING_DIR}/:/training --name my_svm_mode kuberlab/tensorflow-serving:cpu-27-1.5.0 --model_name=my_svm_mode --model_base_path=training
```

## Now lets write Golang code to make remote invocations.

We are using pregenerate protobufs stubs for Tensorflow, unfortunally for now you can't generate it automatically from Tensforflow source code, it is manually moderated process. Plees see [Reference to code](https://kuberlab.com])

```
package main

import (
	"context"
	"fmt"
	"github.com/dreyk/tensorflow-serving-go/pkg/tensorflow/core/example"
	tf "github.com/dreyk/tensorflow-serving-go/pkg/tensorflow/core/framework"
	"github.com/dreyk/tensorflow-serving-go/pkg/tensorflow_serving/apis"
	"google.golang.org/grpc"
	"google.golang.org/grpc/encoding"
	"time"
	"flag"
)

func main() {
	x := flag.Float64("x",0,"X coordinate of example pint")
	y := flag.Float64("y",0,"Y coordinate of example pint")
	modeName := flag.Float64("model","my_svm_mode","Model name")
	flag.Parse()

	conn, err := grpc.Dial("127.0.0.1:9000", grpc.WithInsecure())
	if err != nil {
		panic(err)
	}
	defer conn.Close()
	client := apis.NewPredictionServiceClient(conn)
	tContext, _ := context.WithTimeout(context.Background(), time.Duration(1*time.Minute))
	mspec := &apis.ModelSpec{
		Name: *modeName,
	}
	//Create TensorProto example
	exp := &example.Example{
		Features: &example.Features{
			Feature: map[string]*example.Feature{
				"x": {
					Kind: &example.Feature_FloatList{
						FloatList: &example.FloatList{
							Value: []float32{float32(*x)},
						},
					},
				},
				"y": {
					Kind: &example.Feature_FloatList{
						FloatList: &example.FloatList{
							Value: []float32{float32(*y)},
						},
					},
				},
			},
		},
	}

	//serialize examples to protobuffer
	codec := encoding.GetCodec("proto")
	msg, err := codec.Marshal(exp)
	if err != nil {
		panic(err)
	}
	req := &apis.PredictRequest{
		ModelSpec: mspec,
		Inputs: map[string]*tf.TensorProto{"examples": {
			StringVal: [][]byte{
				msg,
			},
			Dtype: tf.DataType_DT_STRING,
			TensorShape: &tf.TensorShapeProto{
				Dim: []*tf.TensorShapeProto_Dim{
					{
						Size: 1,
					},
				},
			},
		},
		},
	}

	//make prediction requets
	resp, err := client.Predict(tContext, req)
	if err != nil {
		panic(err)
	}
	//print response
	fmt.Printf("%v\n", resp)
}

```

Now we are good to test it:

```
go run --x 21 --y 45 --model my_svm_mode
```

### How about other languages? Do I still need repeat all those things....?

No. For most of cases you could just our proxy that expose tensorflow gRPC as resfull API, and allow to use more user friendly data structures for inputs.

###