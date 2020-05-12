# keras-action-recognition


## train
```sh 
python train.py --input xxx --model xxx
```

```sh
--input 
    frame_clip, frame_flow, clip
  
--model
    T3D-densenet121, T3D-densenet169, C3D, SlowFast, I3D
```

## dataset
data.csv
```sh
path, class
data/0/0001.mp4, 0
data/0/0002.mp4, 0
data/0/0003.mp4, 0
data/1/0001.mp4, 1
data/1/0002.mp4, 1
data/1/0003.mp4, 1
data/2/0001.mp4, 2
data/2/0002.mp4, 2
data/2/0003.mp4, 2
...
```

## model reference
[SlowFast](https://github.com/xuzheyuan624/slowfast-keras)
<br>
[T3D](https://github.com/rekon/T3D-keras)
<br>
[I3D](https://github.com/dlpbc/keras-kinetics-i3d)
