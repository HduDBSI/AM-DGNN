import os
from data.utils import column_info, dict_info
from help import *

_data_file = {"hetrec2011-delicious-2k": "user_taggedbookmarks.dat", \
              "hetrec2011-lastfm-2k": "user_taggedartists.dat", \
              "hetrec2011-movielens-2k-v2": "user_taggedmovies.dat",\
              "douban-books-5k": "user_taggedbooks.txt"}

_least_k = {"hetrec2011-delicious-2k": 15, "hetrec2011-lastfm-2k": 5, "hetrec2011-movielens-2k-v2": 5,"douban-books-5k": 5}


class Hetrec2011:
    def __init__(self, dataroot, dataset, off=0):
        self._dataroot = os.path.join(dataroot, dataset.split('-')[1])  # 按照数据集的名称生成文件名，比如hetrec2011-delicious-2k，会在dataroot下生成一个delicious目录，存放后续文件
        if not os.path.exists(self._dataroot):
            os.mkdir(self._dataroot)  # 创建delicious目录，存放后续产生的数据文件
        self._dataset = dataset
        self._off = off
        _fileroot = os.path.join(dataroot, dataset)
        self._filepath = os.path.join(_fileroot, _data_file[self._dataset])  # 文件路径直到"user_taggedbookmarks.dat"

    def data_preprocess(self):
        print("--------------raw data--------------")  # 原始数据
        _uit_list = get_file_data(self._filepath, 1, 3)
        column_info(_uit_list)

        print("--------------delete tag--------------")  # 去除tag
        _uit_list = delete_tag(_uit_list, _least_k[self._dataset])
        column_info(_uit_list)

        print("--------------dense index--------------")  # 稠密索引
        _uit_list[:, 0] = index_to_dense(_uit_list[:, 0], file_path=self._dataroot + "/user_map.txt")
        _uit_list[:, 1] = index_to_dense(_uit_list[:, 1], file_path=self._dataroot + "/item_map.txt")
        #_uit_list[:, 2] = utils.index2Dense(_uit_list[:, 2], file_path=self._dataroot + "/tag_map.txt")
        column_info(_uit_list)

        print("--------------total interaction--------------")  # user-item交互记录
        _ui_dict = get_dict_from_list(_uit_list, 0, 1)
        dict_info(_ui_dict)

        print("--------------split data--------------")
        self.train_user_items, self.test_user_items = random_split_user_items_dict(_ui_dict, 0.8, 0.0, off=self._off)
        dict_info(self.train_user_items)
        dict_info(self.test_user_items)

        print("--------------change dict--------------")
        change_dict(self.train_user_items, self.test_user_items)
        dict_info(self.train_user_items)
        dict_info(self.test_user_items)

        write_dict(self.train_user_items, os.path.join(self._dataroot, "train.txt"))
        write_dict(self.test_user_items, os.path.join(self._dataroot, "test.txt"))
        # write_dict(val_user_items, os.path.join(file_root, "val.txt"))

        print("--------------train uit--------------")
        self._train_uit_list = get_train_UIT(self.train_user_items, _uit_list)
        self._train_uit_list[:, 2] = index_to_dense(self._train_uit_list[:, 2], file_path=self._dataroot + "/tag_map.txt")
        np.savetxt(self._dataroot + "/user_item_tag.txt", self._train_uit_list, fmt="%d")
        column_info(self._train_uit_list)


if __name__ == "__main__":
    proc = Hetrec2011("/home/yhj/data", "hetrec2011-lastfm-2k", 0.2)
    proc.data_preprocess()