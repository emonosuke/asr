from torch.utils.data import DataLoader
from dataset import SpeechDataset, collate_fn_train, collate_fn_eval

dataset = SpeechDataset(config_path="params.conf")

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_train, num_workers=2)

for data in dataloader:
    print(data)
    print(data["x_batch"].shape, data["seq_lens"].shape, data["labels"].shape)

    break

dataset_eval = SpeechDataset(config_path="params.conf", no_label=True)

dataloader = DataLoader(dataset=dataset_eval, batch_size=4, shuffle=False, collate_fn=collate_fn_eval, num_workers=2)

for data in dataloader:
    print(data)
    print(data["x_batch"].shape, data["seq_lens"].shape)

    break
