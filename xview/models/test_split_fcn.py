from .split_fcn import FCN


def test_can_build_model():
    net = FCN(num_classes=1, num_units=64)
    net.close()
    assert True
