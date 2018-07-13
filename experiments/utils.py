from sacred.observers import MongoObserver
from pandas import Series
from pymongo import MongoClient
from gridfs import GridFS
from tensorflow.python.summary.summary_iterator import summary_iterator
from xview.settings import EXPERIMENT_DB_HOST, EXPERIMENT_DB_USER, EXPERIMENT_DB_PWD,\
    EXPERIMENT_DB_NAME
from xview.datasets import get_dataset
from bson.json_util import dumps
import zipfile
from numpy import array, nan, inf


def load_data(data_config):
    """
    Load the data specified in the data_config dict.
    """
    dataset_params = {key: val for key, val in data_config.items()
                      if key not in ['dataset', 'use_trainset']}
    return get_dataset(data_config['dataset'], dataset_params)


def get_mongo_observer():
    return MongoObserver.create(url='mongodb://{user}:{pwd}@{host}/{db}'.format(
                                    host=EXPERIMENT_DB_HOST, user=EXPERIMENT_DB_USER,
                                    pwd=EXPERIMENT_DB_PWD, db=EXPERIMENT_DB_NAME),
                                db_name=EXPERIMENT_DB_NAME)


def reverse_convert_datatypes(data):
    if isinstance(data, dict):
        if 'values' in data and len(data) == 1:
            return reverse_convert_datatypes(data['values'])
        if 'py/tuple' in data and len(data) == 1:
            return reverse_convert_datatypes(data['py/tuple'])
        if 'py/object' in data and data['py/object'] == 'numpy.ndarray':
            return array(data['values'])
        for key in data:
            data[key] = reverse_convert_datatypes(data[key])
        return data
    elif isinstance(data, list):
        return [reverse_convert_datatypes(item) for item in data]
    elif isinstance(data, str) and data[0] == '[':
        return eval(data)
    return data


class ExperimentData:
    """Loads experimental data from experiments database."""

    def __init__(self, exp_id):
        """Load data for experiment with id 'exp_id'."""
        client = MongoClient('mongodb://{user}:{pwd}@{host}/{db}'.format(
                             host=EXPERIMENT_DB_HOST, user=EXPERIMENT_DB_USER,
                             pwd=EXPERIMENT_DB_PWD, db=EXPERIMENT_DB_NAME))
        self.db = client[EXPERIMENT_DB_NAME]
        self.fs = GridFS(self.db)
        self.record = self.db.runs.find_one({'_id': exp_id})

    def get_record(self):
        """Get sacred record for experiment."""
        return reverse_convert_datatypes(self.record)

    def get_artifact(self, name):
        """Return the produced outputfile with given name as file-like object."""
        if name not in [artifact['name'] for artifact in self.record['artifacts']]:
            raise UserWarning('ERROR: Artifact {} not found'.format(name))

        artifact_id = next(artifact['file_id'] for artifact in self.record['artifacts']
                           if artifact['name'] == name)
        return self.fs.get(artifact_id)

    def get_summary(self, tag):
        """Return pd.Series of scalar summary value with given tag."""
        search = [artifact['name'] for artifact in self.record['artifacts']
                  if 'events' in artifact['name']]
        if not len(search) > 0:
            raise UserWarning('ERROR: Could not find summary file')
        summary_file = search[0]
        tmp_file = '/tmp/summary'
        with open(tmp_file, 'wb') as f:
            f.write(self.get_artifact(summary_file).read())
        iterator = summary_iterator(tmp_file)

        # go through all the values and store them
        step = []
        value = []
        for event in iterator:
            for measurement in event.summary.value:
                if (measurement.tag == tag):
                    step.append(event.step)
                    value.append(measurement.simple_value)
        return Series(value, index=step)

    def get_weights(self):
        filename = next(artifact['name'] for artifact in self.record['artifacts']
                        if 'weights' in artifact['name'])
        return self.get_artifact(filename)

    def dump(self, path):
        """Dump the entire record and it's artifacts as a zip archieve."""
        if not path.endswith('.zip'):
            path = path + '.zip'
        archive = zipfile.ZipFile(path, 'w')
        for artifact in self.record['artifacts']:
            archive.writestr(artifact['name'], self.fs.get(artifact['file_id']).read())
        archive.writestr('record.json', dumps(self.get_record()))

    def update_record(self, changes):
        """Apply changes to the record."""
        self.record.update(changes)
        self.db.runs.replace_one({'_id': self.record['_id']}, self.record)
