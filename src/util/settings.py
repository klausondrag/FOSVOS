import attr


@attr.s
class Settings:
    is_training = attr.ib()
    start_epoch = attr.ib()
    n_epochs = attr.ib()
    avg_grad_every_n = attr.ib()
    snapshot_every_n = attr.ib()
    is_testing_while_training = attr.ib()
    test_every_n = attr.ib()
    batch_size_train = attr.ib()
    batch_size_test = attr.ib()
    is_visualizing_network = attr.ib()
    is_visualizing_results = attr.ib()
    variant = attr.ib()


@attr.s
class OfflineSettings(Settings):
    is_loading_vgg_caffe = attr.ib()


@attr.s
class OnlineSettings(Settings):
    offline_epoch = attr.ib()
