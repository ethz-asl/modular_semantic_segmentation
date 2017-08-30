from sacred.observers import MongoObserver
from pymongo import MongoClient
from xview.settings import EXPERIMENT_DB_HOST, EXPERIMENT_DB_USER, EXPERIMENT_DB_PWD,\
    EXPERIMENT_DB_NAME


def get_mongo_observer():
    return MongoObserver.create(url='mongodb://{user}:{pwd}@{host}/{db}'.format(
                                    host=EXPERIMENT_DB_HOST, user=EXPERIMENT_DB_USER,
                                    pwd=EXPERIMENT_DB_PWD, db=EXPERIMENT_DB_NAME),
                                db_name='xview_experiments')


class ExperimentData:

    def __init__(self, id):
        client = MongoClient('mongodb://{user}:{pwd}@{host}/{db}'.format(
                             host=EXPERIMENT_DB_HOST, user=EXPERIMENT_DB_USER,
                             pwd=EXPERIMENT_DB_PWD, db=EXPERIMENT_DB_NAME))
        self.db = client['xview_experiments']
        self.record = self.db.runs.find_one({})

    def get_experiment(self, id):
        return self.db.runs.find()[0]
