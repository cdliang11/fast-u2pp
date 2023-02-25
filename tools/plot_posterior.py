#!/usr/bin/bash
# Copyright 2022 Horzion Robotics(Binbin Zhang)

import os
import sys

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# plt.rcParams['font.sans-serif'] = ['FangSong', 'KaiTi', 'SimHei']

font_path = sys.argv[1]
post_dir = sys.argv[2]

font = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False


for f in os.listdir(post_dir):
    if f.endswith('.txt'):
        txt_path = os.path.join(post_dir, f)
        pdf_path = os.path.join(post_dir, '{}.pdf'.format(f[:-4]))
        with open(txt_path, 'r', encoding='utf8') as fin:
            subsampling = int(fin.readline().strip())
            tokens = fin.readline().strip().split()
            prob = np.loadtxt(fin)
            assert len(tokens) == prob.shape[1]
            plt.figure(figsize=(60, 25))
            x = np.arange(prob.shape[0]) * subsampling
            for i in range(len(tokens)):
                y = prob[:, i]
                plt.plot(x, y, label=tokens[i])
                if tokens[i] != '<blank>':
                    maxi = np.argmax(y)
                    maxp = y[maxi]
                    plt.text(maxi * subsampling,
                             maxp,
                             '{} {:.3f}'.format(tokens[i], maxp),
                             fontdict=dict(fontsize=24),
                             fontproperties=font)
            x_major_locator = MultipleLocator(subsampling)
            ax = plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)
            plt.tick_params(labelsize=25)
            plt.xlabel('frame', fontsize=30)
            plt.ylabel('postorior', fontsize=30)
            plt.savefig(pdf_path)

