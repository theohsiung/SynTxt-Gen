# SynTxt-Gen
synthetic data generation for text with 3d information

## folders
`txt_test` modify this to txt folder by mostel


`fonts` 70 fonts included


`Gen.py` run this for data generation

## arguments for Gen.py
| args | Description | default |
| --- | --- | --- |
| `--text_dir` | dir for text editing pair from mostel | txt_test |
| `--data_dir` | Data folder name | SynTxt3D_50K_1 |

## Output Data structure
```
SynTxt-Gen/SynTxt3D_50k_1
├── i_s
|    ├── 00000.png
|    ├── 00001.png
|    └── ......
├── mask_3d_s
|    ├── 00000.png
|    ├── 00001.png
|    └── ......
├── mask_3d_t
├── mask_s
|    ├── 00000.png
|    ├── 00001.png
|    └── ......
├── mask_t
├── t_b
|    ├── 00000.png
|    ├── 00001.png
|    └── ......
└── t_f 
     ├── 00000.png
     ├── 00001.png
     └── ......
```

- i_s: styled text a (src) rendering on background image

- mask_3d_s: the mask with normal vector of styled text a (src)

- mask_3d_t: the mask with normal vector of styled text b (tgt)

- mask_s: the binary mask of styled text a (src)

- mask_t: the binary mask of styled text b (tgt)

- t_b: background image w/o any txt rendering

- t_f: styled text b (tgt) rendering on background image
