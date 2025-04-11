### 生成图片的 segmentic mask
这部分主要是为了分割出图像中的texture-less区域，即墙壁、天花板和地板。

- install
```bash
cd preprocess/mask/maskdino/modeling/pixel_decoder/ops/
sh make.sh
```

提供两种语义分割方法：[Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything) ，[MaskDINO](https://github.com/IDEA-Research/MaskDINO)。
- Lang-SAM 是一种基于SAM(segment-anything)的分割方法，通过text prompt 进行分割，这里使用`ceiling.wall.floor`作为prompt 分割出天花板，墙面，地面等区域
- MaskDINO 是一种有监督的语义分割方法，在后续融合过程中，选择`ceiling`,`wall`,`floor`的区域作为mask。

`Lang-SAM` 方法
```python
cd preprocess/mask
python sam.py -i <imgs_dir> -o <output_dir>
```

`MaskDINO` 方法
```python
cd preprocess/mask
python maskdino.py --config maskdino/maskdino_R50_bs16_160k_steplr.yaml \
    --input /path/to/image/**/*.jpg \
    --output /path/to/output
```