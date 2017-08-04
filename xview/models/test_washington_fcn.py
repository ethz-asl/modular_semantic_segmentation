from .washington_fcn import FCN


def test_can_build_model():
    net = FCN(num_classes=1)
    net.close()
