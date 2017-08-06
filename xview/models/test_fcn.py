from .fcn import FCN


def test_can_build_model():
    net = FCN(num_classes=1, num_units=64)
    net.close()
    assert True

def test_can_export_and_load_weights():
    with FCN(num_classes=1, num_units=64) as net:
        path = net.export_weights(save_dir='/tmp/')
        net.load_weights(path)
        assert False
