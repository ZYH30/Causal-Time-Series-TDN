"""Weather-only data factory for the frozen long-term forecasting baselines."""
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Custom


def data_provider(args, flag):
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = flag not in ('test', 'TEST')
    drop_last = False
    batch_size = args.batch_size
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
