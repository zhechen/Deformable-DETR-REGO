# REGO-Deformable DETR

By [Zhe Chen](https://scholar.google.cz/citations?user=Jgt6vEAAAAAJ&hl),  [Jing Zhang](https://scholar.google.com/citations?user=9jH5v74AAAAJ&hl), and [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl).

This repository is the implementation of the paper [Recurrent Glimpse-based Decoder for Detection with Transformer](https://arxiv.org/abs/2112.04632).

## TODO
1. Release paper (finished)

2. Revise README

3. Release results

4. Tidy up codes

5. Double check and reproduce model results

6. Release codes


## Introduction

![deformable_detr](./figs/title.png)

**Abstract.** Although detection with Transformer (DETR) is increasingly popular, its global attention modeling requires an extremely long training period to optimize and achieve promising detection performance. Alternative to existing studies that mainly develop advanced feature or embedding designs to tackle the training issue, we point out that the Region-of-Interest (RoI) based detection refinement can easily help mitigate the difficulty of training for DETR methods. Based on this, we introduce a novel REcurrent Glimpse-based decOder (REGO) in this paper. In particular, the REGO employs a multi-stage recurrent processing structure to help the attention of DETR gradually focus on foreground objects more accurately. In each processing stage, visual features are extracted as glimpse features from RoIs with enlarged bounding box areas of detection results from the previous stage. Then, a glimpse-based decoder is introduced to provide refined detection results based on both the glimpse features and the attention modeling outputs of the previous stage. In practice, REGO can be easily embedded in representative DETR variants while maintaining their fully end-to-end training and inference pipelines. In particular, REGO helps Deformable DETR achieve 44.8 AP on the MSCOCO dataset with only 36 training epochs, compared with the first DETR and the Deformable DETR that require 500 and 50 epochs to achieve comparable performance, respectively. Experiments also show that REGO consistently boosts the performance of different DETR detectors by up to 7% relative gain at the same setting of 50 training epochs.

## License

This project is released under the [Apache 2.0 license](./LICENSE).


## Citing Deformable DETR
If you find REGO useful in your research, please consider citing:
```bibtex
@article{chen2021recurrent,
  title={Recurrent Glimpse-based Decoder for Detection with Transformer},
  author={Chen, Zhe and Zhang, Jing and Tao, Dacheng},
  journal={arXiv preprint arXiv:2112.04632},
  year={2021}
}
```

## Main Results

| <sub><sub>Method</sub></sub>   | <sub><sub>Epochs</sub></sub> | <sub><sub>AP</sub></sub> | <sub><sub>AP<sub>S</sub></sub></sub> | <sub><sub>AP<sub>M</sub></sub></sub> | <sub><sub>AP<sub>L</sub></sub></sub> | <sub><sub>params<br>(M)</sub></sub> | <sub><sub>FLOPs<br>(G)</sub></sub> | <sub><sub>Total<br>Train<br>Time<br>(GPU<br/>hours)</sub></sub> | <sub><sub>Train<br/>Speed<br>(GPU<br/>hours<br/>/epoch)</sub></sub> | <sub><sub>Infer<br/>Speed<br/>(FPS)</sub></sub> | <sub><sub>Batch<br/>Infer<br/>Speed<br>(FPS)</sub></sub> | <sub><sub>URL</sub></sub>                     |
| ----------------------------------- | :----: | :--: | :----: | :---: | :------------------------------: | :--------------------:| :----------------------------------------------------------: | :--: | :---: | :---: | ----- | ----- |
| <sub><sub>Faster R-CNN + FPN</sub></sub> | <sub>109</sub> | <sub>42.0</sub> | <sub>26.6</sub> | <sub>45.4</sub> | <sub>53.4</sub> | <sub>42</sub> | <sub>180</sub> | <sub>380</sub> | <sub>3.5</sub> | <sub>25.6</sub> | <sub>28.0</sub> | <sub>-</sub> |
| <sub><sub>DETR</sub></sub> | <sub>500</sub> | <sub>42.0</sub> | <sub>20.5</sub> | <sub>45.8</sub> | <sub>61.1</sub> | <sub>41</sub> | <sub>86</sub> | <sub>2000</sub> | <sub>4.0</sub> |     <sub>27.0</sub>       |         <sub>38.3</sub>           | <sub>-</sub> |
| <sub><sub>DETR-DC5</sub></sub>      | <sub>500</sub> | <sub>43.3</sub> | <sub>22.5</sub> | <sub>47.3</sub> | <sub>61.1</sub> | <sub>41</sub> |<sub>187</sub>|<sub>7000</sub>|<sub>14.0</sub>|<sub>11.4</sub>|<sub>12.4</sub>| <sub>-</sub> |
| <sub><sub>DETR-DC5</sub></sub>      | <sub>50</sub> | <sub>35.3</sub> | <sub>15.2</sub> | <sub>37.5</sub> | <sub>53.6</sub> | <sub>41</sub> |<sub>187</sub>|<sub>700</sub>|<sub>14.0</sub>|<sub>11.4</sub>|<sub>12.4</sub>| <sub>-</sub> |
| <sub><sub>DETR-DC5+</sub></sub>     | <sub>50</sub> | <sub>36.2</sub> | <sub>16.3</sub> | <sub>39.2</sub> | <sub>53.9</sub> | <sub>41</sub> |<sub>187</sub>|<sub>700</sub>|<sub>14.0</sub>|<sub>11.4</sub>|<sub>12.4</sub>| <sub>-</sub> |
| **<sub><sub>Deformable DETR<br>(single scale)</sub></sub>** | <sub>50</sub> | <sub>39.4</sub> | <sub>20.6</sub> | <sub>43.0</sub> | <sub>55.5</sub> | <sub>34</sub> |<sub>78</sub>|<sub>160</sub>|<sub>3.2</sub>|<sub>27.0</sub>|<sub>42.4</sub>| <sub>[config](./configs/r50_deformable_detr_single_scale.sh)<br/>[log](https://drive.google.com/file/d/1n3ZnZ-UAqmTUR4AZoM4qQntIDn6qCZx4/view?usp=sharing)<br/>[model](https://drive.google.com/file/d/1WEjQ9_FgfI5sw5OZZ4ix-OKk-IJ_-SDU/view?usp=sharing)</sub> |
| **<sub><sub>Deformable DETR<br>(single scale, DC5)</sub></sub>** | <sub>50</sub> | <sub>41.5</sub> | <sub>24.1</sub> | <sub>45.3</sub> | <sub>56.0</sub> | <sub>34</sub> |<sub>128</sub>|<sub>215</sub>|<sub>4.3</sub>|<sub>22.1</sub>|<sub>29.4</sub>| <sub>[config](./configs/r50_deformable_detr_single_scale_dc5.sh)<br/>[log](https://drive.google.com/file/d/1-UfTp2q4GIkJjsaMRIkQxa5k5vn8_n-B/view?usp=sharing)<br/>[model](https://drive.google.com/file/d/1m_TgMjzH7D44fbA-c_jiBZ-xf-odxGdk/view?usp=sharing)</sub> |
| **<sub><sub>Deformable DETR</sub></sub>** | <sub>50</sub> | <sub>44.5</sub> | <sub>27.1</sub> | <sub>47.6</sub> | <sub>59.6</sub> | <sub>40</sub> |<sub>173</sub>|<sub>325</sub>|<sub>6.5</sub>|<sub>15.0</sub>|<sub>19.4</sub>|<sub>[config](./configs/r50_deformable_detr.sh)<br/>[log](https://drive.google.com/file/d/18YSLshFjc_erOLfFC-hHu4MX4iyz1Dqr/view?usp=sharing)<br/>[model](https://drive.google.com/file/d/1nDWZWHuRwtwGden77NLM9JoWe-YisJnA/view?usp=sharing)</sub>                   |
| **<sub><sub>+ iterative bounding box refinement</sub></sub>** | <sub>50</sub> | <sub>46.2</sub> | <sub>28.3</sub> | <sub>49.2</sub> | <sub>61.5</sub> | <sub>41</sub> |<sub>173</sub>|<sub>325</sub>|<sub>6.5</sub>|<sub>15.0</sub>|<sub>19.4</sub>|<sub>[config](./configs/r50_deformable_detr_plus_iterative_bbox_refinement.sh)<br/>[log](https://drive.google.com/file/d/1DFNloITi1SFBWjYzvVEAI75ndwmGM1Uj/view?usp=sharing)<br/>[model](https://drive.google.com/file/d/1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI/view?usp=sharing)</sub> |
| **<sub><sub>++ two-stage Deformable DETR</sub></sub>** | <sub>50</sub> | <sub>46.9</sub> | <sub>29.6</sub> | <sub>50.1</sub> | <sub>61.6</sub> | <sub>41</sub> |<sub>173</sub>|<sub>340</sub>|<sub>6.8</sub>|<sub>14.5</sub>|<sub>18.8</sub>|<sub>[config](./configs/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh)<br/>[log](https://drive.google.com/file/d/1ozi0wbv5-Sc5TbWt1jAuXco72vEfEtbY/view?usp=sharing) <br/>[model](https://drive.google.com/file/d/15I03A7hNTpwuLNdfuEmW9_taZMNVssEp/view?usp=sharing)</sub> |

*Note:*

1. ..
2. ..
3. ..


## Installation

Please follow the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

## Usage
When using REGO, please add '--use_rego' to the training or testing scripts. 

