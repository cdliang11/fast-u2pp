#!/usr/bin/python
# -*- coding: utf-8 -*-
import hashlib
import os
import time
import struct
import numpy as np

from hdfs import InsecureClient


class HdfsCli:

    HDFS_NODES = ['yz-cpu005.hogpu.cc', 'yz-cpu006.hogpu.cc']
    # HDFS_NODES = ['alicpu001.hogpu.cc', 'alicpu002.hogpu.cc'] # aliyun hdfs headnodes

    def __init__(self, user):
        node = self.get_ha_node(self.HDFS_NODES)
        self.user = user
        self.client = InsecureClient('http://' + node + ':50070', user=user)
        self.path = '/user/' + user
        self.prefix = 'hdfs://' + node + ':8020'
        self.node = node + ':50070'

    def get_ha_node(self, hdfs_nodes):
        for node in hdfs_nodes:
            hadoop = InsecureClient('http://' + node + ':50070')
            try:
                hadoop.status('/')
                return node
            except Exception:
                continue
        return ''

    def upload(self, local_folder, data=None):
        if data is None:
            data = str(int(time.time()))
            remote_folder = self.path + '/' + self.__getname(data)
            if self.client.status(remote_folder, strict=False) is None:
                self.client.makedirs(remote_folder, permission=755)
            try:
                ret = self.client.upload(remote_folder, local_folder,
                                         n_threads=5)
                return ret
            except Exception:
                return None

    def upload_file(self, remote_folder, local_file):
        if self.client.status(remote_folder, strict=False) is None:
            self.client.makedirs(remote_folder, permission=755)
        # try:
        #    ret = self.client.upload(remote_folder, local_folder, n_threads=5)
        #    return ret
        # except Exception:
        #    print("upload file failed!!!")
        #    return None
        ret = self.client.upload(remote_folder, local_file, n_threads=5, overwrite=True)
        return ret

    def download(self, remote_folder, local_folder):
        if os.path.exists(local_folder) is None:
            os.makedirs(local_folder) 
            if self.client.status(remote_folder, strict=False) is None:
                return None 
            try:
                ret = self.client.download(remote_folder, local_folder,
                                           n_threads=5)
                return ret
            except Exception:
                return None

    def read(self, path):
        with self.client.read(path) as reader:
            content = reader.read()
            return content

    def read_stream(self, path):
        with self.client.read(path) as f:
            return f

    def list(self, remote_folder):
        ret = self.client.list(remote_folder)
        return ret

    def host(self):
        return self.prefix

    def __getname(self, data):
        m = hashlib.md5()
        m.update(data)
        return m.hexdigest()


if __name__ == '__main__':

    feat_dim = 80
    client = HdfsCli()
    last_frm_num = 0
    fout = open('frm_nums', 'w')
    with open('tmp2.lst') as f:
        for line in f.readlines():
            mxfeats = client.read(line.strip())
            tot_byte_num = len(mxfeats)
            idx = 0
            samples = 0
            times = 0
            while idx < tot_byte_num:
                byte_num = struct.unpack('<I', mxfeats[idx + 4:idx + 8])[0]
                frm_num = int((byte_num - 8) / (feat_dim + 1) / 4)
                fmt = '<' + str(frm_num) + 'i' + str(frm_num * feat_dim) + 'f'
                ali_feats = struct.unpack(fmt, mxfeats[idx + 8:idx + byte_num])
                ali_data = np.array(ali_feats[:frm_num])
                feats_data = np.array(ali_feats[frm_num:])
                feats_data = np.reshape(feats_data, (frm_num, feat_dim))
                idx += (8 + byte_num)  # 4 bytes magic and 4 bytes byte_num
                samples += 1
                print(ali_data)
                # print(feats_data)
                # if frm_num != last_frm_num:
                #    if samples % 10 != 1:
                #        print(last_frm_num, frm_num, samples)
                fout.write(str(frm_num) + '\n')
                last_frm_num = frm_num
                times += 1
    fout.close()
