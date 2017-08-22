from .synthia import Synthia, SYNTHIA_BASEPATH
from os import path


def test_can_find_data():
    data = Synthia(['UNITTEST-SEQUENCE'], 64)
    assert len(data.testset) > 0


def test_preprocessing_produces_all_ouputs():
    Synthia(['UNITTEST-SEQUENCE'], 64, force_preprocessing=True)
    assert path.exists(path.join(SYNTHIA_BASEPATH, 'UNITTEST-SEQUENCE/resized_rgb'))
    assert path.exists(path.join(SYNTHIA_BASEPATH,
                                 'UNITTEST-SEQUENCE/resized_rgb/000000.png'))
    assert path.exists(path.join(SYNTHIA_BASEPATH, 'UNITTEST-SEQUENCE/resized_labels'))
    assert path.exists(path.join(SYNTHIA_BASEPATH,
                                 'UNITTEST-SEQUENCE/resized_labels/000000.npy'))


def test_can_open_data():
    data = Synthia(['UNITTEST-SEQUENCE'], 64)
    test_image = data.testset[0]
    print('test_image: {}'.format(test_image['image_name']))
    blob = data.next()
    # We test that dimensions match between resized image and resized labels.
    assert blob['rgb'].shape[:3] == blob['labels'].shape[:3]
