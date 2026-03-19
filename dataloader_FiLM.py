import torch
import torch.utils.data
import os
import random
import json
import librosa

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, noisy_root, clean_root, cut_len=16000*2):
        self.cut_len = cut_len
        self.noisy_root = noisy_root
        self.clean_root = clean_root
        
        self.text_emb_base = "/home/iasp_guest1/tzx/code/m2d/output/text_embeddings"
        self.audio_emb_base = "/home/iasp_guest1/tzx/code/m2d/output/audio_embeddings"

        self.data_list = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_path = item["path"]
        noise_kind = item["noise_kind"]  
        noisy_name = os.path.basename(full_path)

        noisy_file = os.path.join(self.noisy_root, noisy_name)
        clean_name = noisy_name
        clean_file = os.path.join(self.clean_root, clean_name)

        clean_ds, _ = librosa.load(clean_file, sr=16000)
        noisy_ds, _ = librosa.load(noisy_file, sr=16000)

        clean_ds = torch.tensor(clean_ds).squeeze()
        noisy_ds = torch.tensor(noisy_ds).squeeze()

        length = len(clean_ds)
        assert length == len(noisy_ds)

        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []

            for _ in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)

            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])

            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            wav_start = random.randint(0, length - self.cut_len)
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]

        text_emb_path = os.path.join(self.text_emb_base, f"{noise_kind}_embedding.pth")
        audio_emb_path = os.path.join(self.audio_emb_base, f"{noise_kind}_audio_embedding.pth")

        # [1, 768] -> [768]
        f_t = torch.load(text_emb_path).squeeze()
        
        # 加载音频特征
        f_a = torch.load(audio_emb_path).squeeze()

        return clean_ds, noisy_ds, f_a, f_t, length


def load_data(batch_size, n_cpu, cut_len):
    train_json = "/home/iasp_guest1/tzx/database/dataconvert/dataset/new-55train.jsonl"
    test_json = "/home/iasp_guest1/tzx/database/dataconvert/dataset/new-55test.jsonl"

    train_noisy_root = "/home/iasp_guest1/tzx/database/dataconvert/dataset/new-55train"
    test_noisy_root = "/home/iasp_guest1/tzx/database/dataconvert/dataset/new-55test"

    train_clean_root = "/home/iasp_guest1/tzx/database/dataconvert/dataset/segs5train"
    test_clean_root = "/home/iasp_guest1/tzx/database/dataconvert/dataset/segs5test"

    train_ds = DemandDataset(train_json, train_noisy_root, train_clean_root, cut_len)
    test_ds = DemandDataset(test_json, test_noisy_root, test_clean_root, cut_len)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_cpu
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_cpu
    )

    return train_loader, test_loader