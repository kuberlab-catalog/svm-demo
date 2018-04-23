import argparse
import logging
import os
from mlboardclient.api import client
from mlboardclient.api.v2 import optimizator
import json

logging.basicConfig(
    format='%(asctime)s %(levelname)-10s %(name)-25s [-] %(message)s',
    level='INFO'
)
SUCCEEDED = 'Succeeded'
FAILED = 'Failed'
LOG = logging.getLogger('INFO')


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.debug:
        logging.root.setLevel('DEBUG')

    m = client.Client()
    app = m.apps.get()
    task = app.task('train')
    if args.data_version!='':
        task.config['datasetRevisions']=[{'volumeName': 'data', 'revision': args.data_version}]
    task.resource('worker')['command'] = 'python svm.py --steps=1000 --checkpoint_dir=$TRAINING_DIR/$BUILD_ID'
    spec = (optimizator.ParamSpecBuilder().resource('worker')
            .param('l2_regularization')
            .bounds(1,10)
            .param('l1_regularization')
            .bounds(1,10)
            .build())
    LOG.info('Run with param spec = %s', spec)
    result = task.optimize(
        'accuracy',
        spec,
        init_steps=args.init_steps,
        iterations=args.iterations,
        method=args.method,
        max_parallel=args.parallel,
        direction='maximize'
    )
    best = result['best']
    LOG.info('Found best build %s:%s: %.2f', best.name,best.build,best.exec_info['accuracy'])
    client.update_task_info({'checkpoint_path':best.exec_info['checkpoint_path'],
                             'accuracy':best.exec_info['accuracy'],'build':best.build})
    LOG.info('Exporting model %s...',best.build)
    export = app.task('export')
    export.resource('run')['command'] = 'python svm.py --export_model'
    export.resource('run')['args']= {
        'catalog_name': 'my_svm_model',
        'task_name': best.name,
        'build_id': best.build,
        'checkpoint_dir': best.exec_info['checkpoint_path']
    }
    export.start()
    export.wait()
    client.update_task_info({'model_path':export.exec_info['model_path']})



def get_parser():
    parser = argparse.ArgumentParser(
        description='Calculate steps'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging',
    )
    parser.add_argument(
        '--init_steps',
        type=int,
        default=5,
        help='Number of init steps'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='skopt',
        help='Optimization method'
    )
    parser.add_argument(
        '--data_version',
        type=str,
        default='',
        help='Data version'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='How many task will be executed in paralle'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of iterations'
    )
    return parser

if __name__ == '__main__':
    main()
