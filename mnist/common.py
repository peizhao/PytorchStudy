class Config:
    def __init__(self, epoch, lr, use_cuda = True):
        self.epoch = epoch
        self.lr =lr
        self.use_cuda = use_cuda