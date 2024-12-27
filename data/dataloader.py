import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CollateFn(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        # 将批次数据转换为Tensor
        tensor = torch.from_numpy(np.array(batch)).float()
        return tensor


class SpeakerDataset(Dataset):
    def __init__(self, npy_path):
        # 使用np.load从npy文件加载数据
        self.data = np.load(npy_path)  # 假设npy文件是一个二维数组 (num_speakers, 64)
        
    def __getitem__(self, index):
        # 获取指定索引的说话人表征
        audio_stft = self.data[index]
        return audio_stft

    def __len__(self):
        # 返回数据集大小
        return len(self.data)


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=False):
    # 使用自定义的CollateFn
    _collate_fn = CollateFn()
    
    # 构建DataLoader
    dataLoader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataLoader


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)


"""test"""
if __name__ == '__main__':
    file_path = "c:\\Users\\leeklll\\Documents\\DL\\inano_data\\spk_emb_test.npy"
    dataset = SpeakerDataset(file_path)  # 通过AWdataset类加载数据
    dataloader = get_dataloader(dataset, batch_size=32)

    # 测试Dataloader
    for batch in dataloader:
        print(batch.shape)  # 输出每个批次的形状，例如 (batch_size, 64)
        break
