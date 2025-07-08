from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from prediction.predict import validate
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=collate_fn,
                            pin_memory=args.cuda)
_, yhat = validate(

