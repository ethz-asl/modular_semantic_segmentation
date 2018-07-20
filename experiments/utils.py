from sacred.observers import MongoObserver
from pandas import Series
from pymongo import MongoClient
from gridfs import GridFS
from tensorflow.python.summary.summary_iterator import summary_iterator
import xview.settings as settings
from xview.datasets import get_dataset
from bson.json_util import dumps
import json
import zipfile
from os import path, listdir
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
        """Load data for experiment with id 'exp_id'.

        Follwing the settings, data is either loaded from the mongodb connection or,
        as a fallback, from the specified directory.
        """
        if hasattr(settings, 'EXPERIMENT_DB_HOST') and settings.EXPERIMENT_DB_HOST:
            client = MongoClient('mongodb://{user}:{pwd}@{host}/{db}'.format(
                                host=EXPERIMENT_DB_HOST, user=EXPERIMENT_DB_USER,
                                pwd=EXPERIMENT_DB_PWD, db=EXPERIMENT_DB_NAME))
            self.db = client[EXPERIMENT_DB_NAME]
            self.fs = GridFS(self.db)
            self.record = self.db.runs.find_one({'_id': exp_id})
            self.artifacts = [artifact['name']
                              for artifact in self.record['artifacts']]
        elif hasattr(settings, 'EXPERIMENT_STORAGE_FOLDER') \
                and settings.EXPERIMENT_STORAGE_FOLDER:
            if exp_id in listdir(settings.EXPERIMENT_STORAGE_FOLDER):
                self.exp_path = path.join(settings.EXPERIMENT_STORAGE_FOLDER, exp_id)
                with open(path.join(self.exp_path, 'run.json')) as run_json:
                    record = json.load(run_json)
                with open(path.join(self.exp_path, 'info.json')) as info_json:
                    record['info'] = json.load(info_json)
                with open(path.join(self.exp_path, 'config.json')) as config_json:
                    record['config'] = json.load(config_json)
                with open(path.join(self.exp_path, 'cout.txt')) as captured_out:
                    record['captured_out'] = captured_out.read()
                self.artifacts = listdir(self.exp_path)
            elif '%s.zip' % exp_id in listdir(settings.EXPERIMENT_STORAGE_FOLDER):
                self.zipfile = path.join(settings.EXPERIMENT_STORAGE_FOLDER,
                                         '%s.zip' % exp_id)
                archive = zipfile.ZipFile(self.zipfile)
                record = json.loads(archive.read('run.json'))
                record['info'] = json.loads(archive.read('info.json'))
                record['config'] = json.loads(archive.read('config.json'))
                record['captured_out'] = archive.read('cout.txt')
                archive.close()
                self.artifacts = archive.namelist()
            else:
                raise UserWarning('Specified experiment not found.')
            self.record = record

    def get_record(self):
        """Get sacred record for experiment."""
        return reverse_convert_datatypes(self.record)

    def get_artifact(self, name):
        """Return the produced outputfile with given name as file-like object."""
        if hasattr(self, 'fs'):
            if name not in self.artifacts:
                raise UserWarning('ERROR: Artifact {} not found'.format(name))

            artifact_id = next(artifact['file_id']
                               for artifact in self.record['artifacts']
                               if artifact['name'] == name)
            return self.fs.get(artifact_id)
        elif hasattr(self, 'exp_path'):
            if name not in self.artifacts:
                raise UserWarning('ERROR: Artifact {} not found'.format(name))
            with open(path.join(self.exp_path, name)) as artifact:
                return artifact
        else:
            if name not in self.artifacts:
                raise UserWarning('ERROR: Artifact {} not found'.format(name))
            archive = zipfile.ZipFile(self.zipfile)
            with archive.open(name) as artifact:
                return artifact

    def get_summary(self, tag):
        """Return pd.Series of scalar summary value with given tag."""
        search = [artifact for artifact in self.artifacts if 'events' in artifact]
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
        filename = next(artifact for artifact in self.artifacts if 'weights' in artifact)
        return self.get_artifact(filename)

    def dump(self, path):
        """Dump the entire record and it's artifacts as a zip archieve."""
        if not path.endswith('.zip'):
            path = path + '.zip'
        archive = zipfile.ZipFile(path, 'w')
        for artifact in self.record['artifacts']:
            archive.writestr(artifact['name'], self.fs.get(artifact['file_id']).read())
        # following the FileStorageObserver, we need to create different files for config,
        # output, info and the rest of the record
        record = self.get_record()
        archive.writestr('config.json', dumps(record['config']))
        archive.writestr('cout.txt', record['captured_out'])
        archive.writestr('info.json', dumps(record['info']))
        record.pop('config', None)
        record.pop('captured_out', None)
        record.pop('info', None)
        archive.writestr('run.json', dumps(record))
        archive.close()

    def update_record(self, changes):
        """Apply changes to the record."""
        # so far only implemented for database version
        assert hasattr(self.db)
        self.record.update(changes)
        self.db.runs.replace_one({'_id': self.record['_id']}, self.record)
