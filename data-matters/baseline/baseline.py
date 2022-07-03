import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, asdict
import objectpath


@dataclass
class Biomarker:
    participant: str
    name: str
    value: float
    time: np.datetime64 = None
    description: str = ''
    arm: str = ''
    targeted_date: np.datetime64 = None
    enrolled_date: np.datetime64 = None
    baseline_targeted_days: np.datetime64 = None
    baseline_enrolled_days: np.datetime64 = None

    def __post_init__(self):

        # set baseline from targeted
        if self.__dict__.get('targeted_date') != None:
            time = self.__dict__.get('time')
            target_date = self.__dict__.get('targeted_date')
            _baseline_time = self._baseline_time(target_date, time)
            self.__dict__.update({'baseline_targeted_days': _baseline_time})

        # set baseline from enrolled
        if self.__dict__.get('enrolled_date') != None:
            time = self.__dict__.get('time')
            target_date = self.__dict__.get('enrolled_date')
            _baseline_time = self._baseline_time(target_date, time)
            self.__dict__.update({'baseline_enrolled_days': _baseline_time})

        return self

    @classmethod
    def _baseline_time(cls, targeted_date, time):

        delta = np.datetime64(time) - np.datetime64(targeted_date)
        return delta.astype('d')

    def _update(self, key, value):

        if key in self.__dict__.keys():
            self.__dict__.update({key: value})
        return self


@dataclass()
class Participant:
    name: str

    def __post_init__(self):
        setattr(self, 'biomarkers', dict())
        setattr(self, 'biomarkers_class', dict())
        self.biomarker_keys = self.biomarkers_class.keys()

    def _add_measurement(self, marker):

        ## check to see if biomarker already in dict
        if marker.name in self.biomarkers.keys():
            # ensure that the participant name matches
            assert (marker.participant == self.name), f'participant {marker.participant} does not match {self.name}'

            # ensure only unique biomarkers are added
            for biomark in self.biomarkers_class.get(marker.name):
                assert (marker != biomark), 'biomarker exists'

            ## old data point
            bm_v = self.biomarkers.get(marker.name)
            bm_v_cl = self.biomarkers_class.get(marker.name)

            ## set new value
            self.biomarkers[marker.name] = bm_v.__add__([asdict(marker)])
            self.biomarkers_class[marker.name] = bm_v_cl.__add__([marker])

        else:
            # update the biomarkers dict with a new key
            self.biomarkers.update({marker.name: [asdict(marker)]})
            self.biomarkers_class.update({marker.name: [marker]})

        return self

    def bio_query(self, query, _class=False):

        # get biomarker data
        if _class:
            tree = objectpath.Tree(self.biomarkers_class)
        else:
            tree = objectpath.Tree(self.biomarkers)

        # query tree w/ object path
        return tuple(tree.execute(query))

    def as_dataframe(self):
        df_list = [pd.DataFrame(self.biomarkers[x]) for x in list(self.biomarker_keys)]
        return pd.concat(df_list)


@dataclass
class Study:
    name: str

    def __post_init__(self):
        setattr(self, 'data', dict())
        setattr(self, 'participants', dict())
        setattr(self, 'participants_class', dict())
        self.data = self.data
        self.member_keys = self.data.keys()

        return self

    def _add_participant(self, Participant):
        # ensure that the participant name matches
        # ensure only unique participants are added
        if Participant.name in self.participants_class.keys():
            print(f'participant {Participant.name} exists in {self.name}')
            pass
        else:
            self.data.update({Participant.name: [Participant.biomarkers]})
            self.participants.update({Participant.name: [Participant]})
            self.participants_class.update({Participant.name: [Participant]})

        return self

    def _add_participants(self, list_of_part):
        for part in list_of_part:
            self._add_participant(part)

        return self

    def bio_query(self, query, as_dataframe=False, override=False):
        # get biomarker data
        if override:
            tree = objectpath.Tree(override)
        else:
            tree = objectpath.Tree(self.data)

        if as_dataframe:
            return pd.DataFrame(tuple(tree.execute(query)))

        else:
            return tuple(tree.execute(query))

    def as_dataframe(self):
        df_list = []
        for part in self.participants.keys():
            df_list.append(self.participants[part][0].as_dataframe())
        return pd.concat(df_list)


class StudyStats(Study):

    def __init__(self, Study):
        self.study = Study

    @classmethod
    def _ci(cls, tuples, low=2.5, high=97.5):
        _values = np.array(tuples)[~np.isnan(tuples)]
        return [np.percentile(_values, [low, high])]

    @classmethod
    def _cat_count(cls, tuples):
        _values = np.array(tuples)[~np.isnan(tuples)]
        unique, counts = np.unique(_values, return_counts=True)
        return dict(zip(unique, counts))

    @classmethod
    def _mode(cls, tuples):
        _values = np.array(tuples)[~np.isnan(tuples)]

        return stats.mode(_values)

    @classmethod
    def _gap_frac(cls, tuples):
        _values = np.array(tuples)[~np.isnan(tuples)]

        # unique, counts
        return np.unique(_values, return_counts=True)

    @classmethod
    def _mean(cls, tuples):
        _values = np.array(tuples)[~np.isnan(tuples)]
        return _values.mean()

    @classmethod
    def _median(cls, tuples):
        _values = np.array(tuples)[~np.isnan(tuples)]

        return np.median(_values)

    @classmethod
    def _std(cls, tuples):
        _values = np.array(tuples)[~np.isnan(tuples)]

        return _values.std()
