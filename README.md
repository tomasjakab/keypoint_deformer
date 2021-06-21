# [KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://tomasjakab.github.io/KeypointDeformer/)

[Tomas Jakab](http://www.robots.ox.ac.uk/~tomj), Richard Tucker, Ameesh Makadia, Jiajun Wu, Noah Snavely, Angjoo Kanazawa.
CVPR, 2021 (Oral presentation).

We present KeypointDeformer, a novel unsupervised method for shape control through automatically discovered 3D keypoints. Our approach produces intuitive and semantically consistent control of shape deformations. Moreover, our discovered 3D keypoints are consistent across object category instances despite large shape variations. Since our method is unsupervised, it can be readily deployed to new object categories without requiring expensive annotations for 3D keypoints and deformations.


## Install
Clone the repo
```
git clone https://github.com/tomasjakab/keypoint_deformer
cd keypoint_deformer
```

Install using [conda](https://docs.conda.io/en/latest/):
```
conda env create -f environment.yml 
conda activate keypointdeformer
```
Set-up python path:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Training
Download [ShapeNet](https://shapenet.org/download/shapenetcore) to `data/shapenet`. The path to ShapeNet can be also customized in config files `configs/*` with the option `mesh_dir`.

To train a model on the airplane category with 8 unsupervised keypoints run: 
```
python scripts/main.py -c configs/airplane-8kpt.yaml
```

To train a model on the chair category with 12 unsupervised keypoints run:
```
python scripts/main.py -c configs/chair-12kpt.yaml
```

## Testing
To test the trained model run:
```
python scripts/main.py -c configs/airplane-8kpt.yaml -t configs/test.yaml 
```
This will create result files in `logs/airplane-8kpt/test/<SAMPLE NAME>`. The file `source_mesh.obj` contains the input mesh and the file `source_keypoints.txt` predicted unsupervised keypoints. 

To visualize the results run:
```
python browse3d/browse3d.py --log_dir logs/airplane-8kpt/test --port 5050
```
and open `localhost:5050` in your web browser.

## Demo
Try the [interactive demo](https://tomasjakab.github.io/KeypointDeformer/demo.html) without any instalation.


## Acknowledgments
Parts of the code are based on [Neural Cages](https://github.com/yifita/deep_cage).
