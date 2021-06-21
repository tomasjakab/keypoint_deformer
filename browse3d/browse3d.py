import json
import os
import random

import bottle
import configargparse
from bottle import route, run, static_file, template

bottle.TEMPLATE_PATH.insert(0, os.path.join('browse3d', 'views'))
# bottle.debug(True)

app_root = os.path.dirname(os.path.abspath(__file__))

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')
parser.add_argument('-V', "--visuals", type=str, default=os.path.join('browse3d', 'configs', 'default.json'), help='')
parser.add_argument('-l', "--log_dir", type=str, default='', help="")
parser.add_argument("--im_size", type=int, default=(200, 200), nargs=2, help="")
parser.add_argument("--n_samples", type=int, default=10, help="")
parser.add_argument("--samples", type=int, nargs='+', default=None, help="")
parser.add_argument("--port", type=int, default=5050, help="")
parser.add_argument("--vertical", action='store_true', help="")
parser.add_argument("--random", action='store_true', help="")
parser.add_argument("--seed", type=int, default=0, help="")
opt = parser.parse_args()

log_dir = opt.log_dir
print(log_dir)

@route('/')
@route('/rel/<path:path>')
def main(**args):
    with open(opt.visuals, 'r') as f:
        visuals = json.load(f)

    if 'path' in args:
        log_dir_dym = os.path.join(log_dir, args['path'])
    else:
        log_dir_dym = log_dir
    samples = os.listdir(log_dir_dym)
    samples = [x for x in samples if os.path.isdir(os.path.join(log_dir_dym, x))]
    samples = sorted(samples)

    if opt.samples is not None:
        samples = [samples[i] for i in opt.samples]
    if opt.random:
        random.seed(opt.seed)
        random.shuffle(samples)
    samples = samples[:min(len(samples), opt.n_samples)]

    return template('browse3d', title=opt.log_dir, samples=samples, opt=opt, visuals=visuals)

# serve data
@route('/data/<path:path>')
def serve_data(path):
    return static_file(path, root=log_dir)

# serve js
@route('/js/<file>')
def serve_js(file):
    return static_file(file, root=os.path.join(app_root, 'www', 'js'))


run(host='localhost', port=opt.port, debug=True)
