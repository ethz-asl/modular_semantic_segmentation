from .split_fcn import SplitFCN
from xview.datasets.synthia import Synthia


def test_can_build_model():
    net = SplitFCN(num_classes=1, num_units=64)
    net.close()
    assert True


def test_can_run_training():
    data = Synthia(['UNITTEST-SEQUENCE'], 2)
    config = {'num_classes': 14,
              'num_units': 5,
              'dropout_rate': 0.2,
              'learning_rate': 0.01,
              'batch_normalization': False}
    with SplitFCN(**config) as net:
        net.fit(data, 1)
