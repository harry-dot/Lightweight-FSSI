from tensorboardX import SummaryWriter


class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardWriter, self).__init__(logdir)

    def log_training(self, train_loss, intra_loss,inter_loss,step):
        self.add_scalar('classification_loss', train_loss, step)
        self.add_scalar('intra_class_loss',intra_loss,step)
        self.add_scalar('inter_class_loss',inter_loss,step)