import argparse
import logging
import os
from mlboardclient.api import client
import json
import pandas as pd
import time

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
    if args.task!='':
        app = m.apps.get()
        serving = app.servings[0]
        srv = serving.start(args.task, args.build)
        LOG.info("Start serving")
        name = os.environ.get('PROJECT_NAME') + '-' + srv.config['name'] + '-' +args.task + '-' + args.build
        LOG.info("Waiting serving ready...")
        status = 0
        for i in range(6):
            time.sleep(10)
            resp = m.servings.call(name,'svm',{
                "features": [
                    {"x": {"Float": 0},"y": {"Float": 0}}
                ]},port='9000')
            status = resp.status_code
            if status==200:
                break
        if status!=200:
            srv.stop()
            raise RuntimeError('Failed start serving')
        res = validate(m,name)
        srv.stop()
        if res!='':
            raise RuntimeError(res)
    else:
        res = validate(m,args.model)
        if res!='':
            raise RuntimeError(res)


def validate(m,name):
    LOG.info("Start validation...")
    data = pd.read_csv(os.environ.get('DATA_DIR')+"/test.csv")
    all = 0.0
    good = 0.0
    for index, row in data.iterrows():
        x = float(row['x'])
        y = float(row['y'])
        label = int(row['label'])
        resp = m.servings.call(name,'svm',{
            "features": [
                {"x": {"Float": x},"y": {"Float": y}}
            ]},port='9000')
        if resp.status_code == 200:
            res = json.loads(resp.content)['classes'][0]
            if res==label:
                good += 1
            all += 1
        else:
            return 'Failed request to serving: %d' % (resp.status_code)
    accuracy = good/all
    LOG.info("Accuracy %.2d",accuracy)
    client.update_task_info({'accuracy':accuracy})
    return ''

def get_parser():
    parser = argparse.ArgumentParser(
        description='Validate models'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging',
    )
    parser.add_argument(
        '--build',
        type=str,
        default='',
        help='Build'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='',
        help='Task'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='',
        help='Task'
    )
    return parser

if __name__ == '__main__':
    main()
