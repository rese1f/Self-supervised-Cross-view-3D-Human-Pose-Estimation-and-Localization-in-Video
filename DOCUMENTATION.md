# Documentation
This guide includes **all** important information about the whole project and includes the essential principles of the code logic. Please be **sure** that you’ve read `README.md` thoroughly before you move on.

## Overall Layout of the Project

Firstly please read through the tree graph of the whole project to get a basic concept of the whole program.

```shell
.
├── CONTRIBUTING.md
├── DATASETS.md
├── DOCUMENTATION.md
├── DataGenerators
│   ├── __pycache__
│   │   ├── arguments.cpython-38.pyc
│   │   └── data_utils.cpython-38.pyc
│   ├── arguments.py
│   ├── camera_utils.py
│   ├── data
│   │   ├── data_3d_h36m.npz
│   │   └── h36m
│   ├── data_3d_h36m.npz
│   ├── data_utils.py
│   ├── output
│   │   └── data_multi_2d_h36m.npz
│   ├── prepare_dataset.py
│   ├── run.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── random_function.cpython-38.pyc
│   │   ├── camera.py
│   │   ├── collision.py
│   │   ├── cover.py
│   │   ├── random_function.py
│   │   └── seq_collision_eli.py
│   └── visualize.py
├── INFERENCE.md
├── LICENSE
├── README.md
├── VideoPose3D
│   ├── LICENSE
│   ├── checkpoint
│   │   └── pretrained_h36m_cpn.bin
│   ├── common
│   │   ├── arguments.py
│   │   ├── generators.py
│   │   ├── model.py
│   │   └── regressor.py
│   └── run.py
├── configs.ini
└── star
    ├── __init__.py
    ├── config.py
    ├── pytorch
    │   ├── __init__.py
    │   ├── star.py
    │   ├── utils.py
    │   └── verts.py
    └── star_1_1
        ├── LICENSE.txt
        ├── female
        │   └── model.npz
        ├── male
        │   └── model.npz
        └── neutral
            └── model.npz

16 directories, 193 files
```

To make it more illustrative, we’ve elaborated on the illustrations and published it here.

