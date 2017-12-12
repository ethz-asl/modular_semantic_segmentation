from .adapnet import Adapnet
from xview.datasets.synthia import Synthia

config = {'num_classes': 14,
          'num_units': 20,
          'learning_rate': 0.01,
          'num_channels': 3,
          'modality': 'rgb'}


def test_can_build_model():
    net = Adapnet(**config)
    net.close()
    assert True


def test_can_run_training():
    data = Synthia(['UNITTEST-SEQUENCE'], 2)

    with Adapnet(**config) as net:
        net.fit(data, 1)
