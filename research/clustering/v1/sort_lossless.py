"""
Quick script for sorting lossless dicts
"""

import dill as pickle
with open('data/flop_lossy.pkl', 'rb') as file:
    flop_lossy = pickle.load(file)

with open('data/turn_lossy.pkl', 'rb') as file:
    turn_lossy = pickle.load(file)

with open('data/river_lossy.pkl', 'rb') as file:
    river_lossy = pickle.load(file)

things = {'flop': flop_lossy, 'turn': turn_lossy, 'river': river_lossy}

for each in things.keys():
    sorted_lossy = {}
    for k, v in things[each].items():
        private = list(sorted(list(k[:2]), reverse=True))
        public = list(sorted(list(k[2:]), reverse=True))
        key = tuple(private+public)
        sorted_lossy[key] = v


    with open(f'../blueprint_algo/{each}_lossy_2.pkl', 'wb') as file:
        pickle.dump(sorted_lossy, file)
