import argparse
import logging
import os
from functools import partial
from mlboardclient.api import client

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

    ml = client.Client()

    app = ml.apps.get()

    def _task(i,task):
        step_count = 100+i*300
        model = 'svm-s_%d-l1_%d-l2_%d' % (step_count,1,10)
        task.comment = model
        task.resource('worker')['args']= {
            'l2_regularization':10,
            'l1_regularization': 1,
            'steps': step_count,
            'checkpoint_dir': '%s/%s-%s' % (os.environ.get('TRAINING_DIR'),os.environ.get('BUILD_ID'),model)
        }
    def jobs(num):
        for i in range(num):
            yield partial(_task,i)

    task = app.task('train')
    results = task.parallel_run(2, jobs(args.runs))
    best_accuracy = 0
    best = None
    for r in results:
        if r.exec_info['accuracy']>best_accuracy:
            best_accuracy = r.exec_info['accuracy']
            best = r
    LOG.info('BEST %s with accuracy=%.2f',best.exec_info['model_path'],best_accuracy)

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
        '--runs',
        type=int,
        default=1,
        help='Number of runs'
    )

    return parser

if __name__ == '__main__':
    main()
