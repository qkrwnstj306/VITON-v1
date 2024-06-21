# Manual

- Time-and-Head integrated attribution map for generated image: Attribution map for generated image on validation dataset over time step and attention head

'''
python attention_inference.py
'''

- Time integrated attribution map: Attribution map for generated image on validation dataset over time


'''
python attention_inference.py --per_head True
'''

- Head integrated attribution map: Attribution map for generated image on validation dataset over head


'''
python attention_inference.py --per_time True 
'''

- If you want to visualize only one data in validataion dataset

'''
python attenntion_inference.py --only_one_data True --certain_data_idx [image.jpg]

Default: --certain_data_idx 00273_00.jpg
'''
