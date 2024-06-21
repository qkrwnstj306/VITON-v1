# Manual

## Setting

- Dataset: VITON-HD dataset
- - You have to edit code in `dataset.py`: self.data_dir = '../dataset/zalando-hd-resized'

## Attribution map for generated image

- Time-and-Head integrated attribution map: Attribution map for generated image on validation dataset over time step and attention head

```
python attention_inference.py --save_dir ./outputs/{dir_name}
```

- Time integrated attribution map: Attribution map for generated image on validation dataset over time


```
python attention_inference.py --per_head True --save_dir ./outputs/{dir_name}
```

- Head integrated attribution map: Attribution map for generated image on validation dataset over head


```
python attention_inference.py --per_time True --save_dir ./outputs/{dir_name}
```

- If you want to visualize only one data in validation dataset

```
python attenntion_inference.py --only_one_data True --certain_data_idx {image.jpg} --save_dir ./outputs/{dir_name}

Default: --certain_data_idx 00273_00.jpg
```

## Attribution map for reference image

- Time-and-head integrated attribution map: Attribution map for reference image on validation dataset over time step and attention head

```
python attention_inference.py --generated_image False --save_dir ./outputs/{dir_name}
```

- Time integrated attribution map: Attribution map for generated image on validation dataset over time

```
python attention_inference.py --generated_image False --per_head True --save_dir ./outputs/{dir_name}
```

- Head integrated attribution map: Attribution map for generated image on validation dataset over head

```
python attention_inference.py --generated_image False --per_time True --save_dir ./outputs/{dir_name}
```

- Specific-reference attribution map: Attribution map for reference image from which information was extracted when creating a specific patch in the generated image

```
python attention_inference.py --generated_image False --specific_reference_attribution_maps True --patch_index {0-14} --save_dir ./outputs/{dir_name}

patch_index: which cross-attention layer of U-Net will be visualized, (Recommendation Index: 6, 7)
```

- If you want to visualize only one data in validation dataset

```
--only_one_data True --certain_data_idx {img.jpg}
```

