class MockedSequentialBatchSampler:
    """
    MockedSequentialBatchSampler
    """

    def __init__(self, train_ds, micro_num):
        self.train_ds = train_ds
        self.micro_num = micro_num

        self.batch_count = 0
        self.num_consumed_samples_in_epoch = 0

    def __iter__(self):
        num_samples = len(self.train_ds)
        for start in range(0, num_samples, self.micro_num):
            end = min(start + self.micro_num, num_samples)
            self.batch_count += 1
            self.num_consumed_samples_in_epoch += end - start
            yield list(range(start, end))

    def __len__(self):
        return (len(self.train_ds) + self.micro_num - 1) // self.micro_num

    def state_dict(self):
        states = {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
        }
        return states

    def load_state_dict(self, states):
        self.batch_count = states["batch_count"]
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]

    def copy(self):
        copy_sampler = MockedSequentialBatchSampler(self.train_ds, self.micro_num)
        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler
