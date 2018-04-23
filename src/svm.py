import tensorflow as tf
import argparse
from tensorflow import logging
from mlboardclient.api import client

def get_parser():
    parser = argparse.ArgumentParser(
        description='Train SVM model'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging',
    )
    parser.add_argument(
        '--export_model',
        action='store_true',
        help='Export model',
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export model',
    )
    parser.add_argument(
        '--task_name',
        default='',
        help='Task name to export',
    )
    parser.add_argument(
        '--build_id',
        default='',
        help='Build id to export',
    )
    parser.add_argument(
        '--catalog_name',
        type=str,
        default=None,
        help='Publish to catalog',
    )
    parser.add_argument(
        '--l2_regularization',
        type=float,
        default=10.0,
        help='L2-regularization parameter'
    )
    parser.add_argument(
        '--l1_regularization',
        type=float,
        default=0.0,
        help='L1-regularization parameter'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of steps'
    )
    parser.add_argument(
        '--checkpoint_dir',
        required=True,
        help='Checkpoint dir',
    )
    parser.add_argument(
        '--dataset_dir',
        default='/notebooks/data',
        help='Dataset dir',
    )
    return parser

def export(checkpoint_dir,task_name,build_id,catalog_name):
    featureX = tf.contrib.layers.real_valued_column('x')
    featureY = tf.contrib.layers.real_valued_column('y')
    model = tf.contrib.learn.SVM(example_id_column='i',
                                 feature_columns=[featureX, featureY],
                                 model_dir=checkpoint_dir)
    feature_spec = {'x': tf.FixedLenFeature(dtype=tf.float32,shape=[1]),
                    'y': tf.FixedLenFeature(dtype=tf.float32,shape=[1])}
    serving_fn = tf.contrib.learn.utils.input_fn_utils.build_parsing_serving_input_fn(feature_spec)
    export_path = model.export_savedmodel(export_dir_base=checkpoint_dir,serving_input_fn=serving_fn)
    export_path = export_path.decode("utf-8")
    logging.info("\nModel Path: %s",export_path)
    client.update_task_info({'model_path':export_path},task_name=task_name, build_id=build_id)
    client.update_task_info({'model_path':export_path,'checkpoint_path':checkpoint_dir})
    if catalog_name is not None:
        ml = client.Client()
        ml.model_upload(catalog_name,'1.0.' + build_id,export_path)


def train(dataset_dir,checkpoint_dir,l1_regularization,l2_regularization,steps,needExport):
    train_set = tf.data.TextLineDataset(dataset_dir+'/train.csv')
    validation_set = tf.data.TextLineDataset(dataset_dir+'/test.csv')
    logging.info("start build svm model")
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
        client.update_task_info({'accuracy':float(metrics['accuracy']),'loss': float(metrics['loss'])})
    if needExport:
        feature_spec = {'x': tf.FixedLenFeature(dtype=tf.float32,shape=[1]),
                        'y': tf.FixedLenFeature(dtype=tf.float32,shape=[1])}
        #serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        serving_fn = tf.contrib.learn.utils.input_fn_utils.build_parsing_serving_input_fn(feature_spec)
        export_path = model.export_savedmodel(export_dir_base=checkpoint_dir,serving_input_fn=serving_fn)
        export_path = export_path.decode("utf-8")
        logging.info("\nModel Path: %s",export_path)
        client.update_task_info({'model_path':export_path})


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
    if not tf.gfile.Exists(args.checkpoint_dir):
        tf.gfile.MakeDirs(args.checkpoint_dir)
    if args.export_model:
        export(args.checkpoint_dir,args.task_name,args.build_id,args.catalog_name)
    else:
        client.update_task_info({'checkpoint_path':args.checkpoint_dir})
        train(args.dataset_dir,args.checkpoint_dir,args.l1_regularization,args.l2_regularization,args.steps,args.export)

if __name__ == '__main__':
    main()