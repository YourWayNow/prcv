## T2I Benches

0. 配置

0.0. 安装依赖

```bash
pip install python==3.10.19
pip install -r requirements.txt
```

0.1. 数据格式

首先要有一个图像文件夹如 `clipscore/example/images`，里面放要评测的图像文件。

然后要有一个 image name to prompt 的 json 文件如 `clipscore/example/good_captions.json`，格式如下：

```json
{
  "image1": "a cat sitting on a mat.",
  "image2": "a dog playing with a ball."
}
```

### 评测方法

1. Clip Score
    * `cd clipscore && python clipscore.py example/good_captions.json example/images/`

2. LAION-Aesthetics_Predictor V1
    * `cd aesthetic-predictor && python batch_inference.py --image_dir ../clipscore/example/images`

3. ImageReward
    * `cd ImageReward && python batch_inference.py --image_dir ../clipscore/example/images --prompt_dict ../clipscore/example/good_captions.json`
