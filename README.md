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
├── i_s  # src img with different text color and bkg
|    ├── 00000.png
|    ├── 00001.png
|    └── ......
├── mask_3d_s # src img mask with normal vector
|    ├── 00000.png
|    ├── 00001.png
|    └── ......
├── mask_3d_t # tgt img mask with normal vector
├── mask_s # black&white src img mask
|    ├── 00000.png
|    ├── 00001.png
|    └── ......
├── mask_t # black&white tgt img mask
├── t_b # bkg img without text
|    ├── 00000.png
|    ├── 00001.png
|    └── ......
└── t_f # tgt img (Ground Truth)
     ├── 00000.png
     ├── 00001.png
     └── ......
```
