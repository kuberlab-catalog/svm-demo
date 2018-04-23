from mlboardclient.api import client
import numpy as np
import pandas as pd
from numpy import random
import os

import argparse


def main():
    parser = get_parser()
    args = parser.parse_args()
    dataset = '/tmp/data'
    train = pd.DataFrame(random.randint(low=0, high=100, size=(1000,2)),columns=['x', 'y'])
    train['label'] = train.apply(lambda v: 0 if v['x']>v['y']+(5-random.random_sample()*10) else 1,axis=1)
    test = pd.DataFrame(random.randint(low=0, high=100, size=(100,2)),columns=['x', 'y'])
    test['label'] = test.apply(lambda v: 0 if v['x']>v['y'] else 1,axis=1)
    train.to_csv(dataset+'/train.csv')
    test.to_csv(dataset+'/test.csv')
    kl = client.Client()
    kl.datasets.push(os.environ.get('WORKSPACE_NAME'),args.dataset,args.version,dataset,create=True)
    client.update_task_info({'dataset':'%s:%s' % (args.dataset,args.version)})


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='DataSet name',
        dest='dataset',
        type=str,
        default='svm-demo',
    )
    parser.add_argument(
        '--version',
        help='Dataset version',
        dest='version',
        type=str,
        default='1.0.'+os.environ.get('BUILD_ID'),
    )
    return parser

if __name__ == '__main__':
    main()