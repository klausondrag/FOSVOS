import attr


@attr.s
class Settings:
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


@attr.s
class ParentSettings(Settings):
    is_loading_vgg_caffe = attr.ib()


@attr.s
class OnlineSettings(Settings):
    parent_name = attr.ib()
    parent_epoch = attr.ib()
