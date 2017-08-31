from sacred.observers import MongoObserver
from pymongo import MongoClient
from gridfs import GridFS
from xview.settings import EXPERIMENT_DB_HOST, EXPERIMENT_DB_USER, EXPERIMENT_DB_PWD,\
    EXPERIMENT_DB_NAME


def get_mongo_observer():
    return MongoObserver.create(url='mongodb://{user}:{pwd}@{host}/{db}'.format(
                                    host=EXPERIMENT_DB_HOST, user=EXPERIMENT_DB_USER,
                                    pwd=EXPERIMENT_DB_PWD, db=EXPERIMENT_DB_NAME),
                                db_name='xview_experiments')


class ExperimentData:

    def __init__(self, exp_id):
        client = MongoClient('mongodb://{user}:{pwd}@{host}/{db}'.format(
                             host=EXPERIMENT_DB_HOST, user=EXPERIMENT_DB_USER,
                             pwd=EXPERIMENT_DB_PWD, db=EXPERIMENT_DB_NAME))
        self.db = client['xview_experiments']
        self.fs = GridFS(self.db)
        self.record = self.db.runs.find_one({'_id': exp_id})

    def get_record(self):
        return self.record

    def get_artifact(self, name):
        if name not in [artifact['name'] for artifact in self.record['artifacts']]:
            raise UserWarning('ERROR: Artifact {} not found'.format(name))

        artifact_id = next(artifact['file_id'] for artifact in self.record['artifacts']
                           if artifact['name'] == name)
        return self.fs.get(artifact_id)

