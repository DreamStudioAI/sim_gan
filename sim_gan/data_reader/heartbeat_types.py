"""Util functions to process the data."""
from enum import Enum


OTHER_HEART_BEATS = "Others"

NUMBER_OF_HEART_BEATS_TRAIN = {"N": 45852, "S": 944, "V": 3788, "F": 414, "Q": 1505}
NUMBER_OF_HEART_BEATS_TEST = {"N": 44242, "S": 1837, "V": 3220, "F": 388, "Q": 1466}


class MitBihArrhythmiaHeartBeatTypes(Enum):
    """MIT-BIH Arrhythmia heartbeats annotations.

    The annotations are defined here: https://archive.physionet.org/physiobank/annotations.shtml

    Code		Description
      N		Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
      L		Left bundle branch block beat
      R		Right bundle branch block beat
      B		Bundle branch block beat (unspecified)
      A		Atrial premature beat
      a		Aberrated atrial premature beat
      J		Nodal (junctional) premature beat
      S		Supraventricular premature or ectopic beat (atrial or nodal)
      V		Premature ventricular contraction
      r		R-on-T premature ventricular contraction
      F		Fusion of ventricular and normal beat
      e		Atrial escape beat
      j		Nodal (junctional) escape beat
      n		Supraventricular escape beat (atrial or nodal)
      E		Ventricular escape beat
      /		Paced beat
      f		Fusion of paced and normal beat
      Q		Unclassifiable beat
      ?		Beat not classified during learning

    """
    N = 0
    L = 1
    R = 2
    e = 3
    j = 4
    A = 5
    a = 6
    J = 7
    S = 8
    V = 9
    E = 10
    F = 11
    Q = 12


class AAMIHeartBeatTypes(Enum):
    """AAMI heartbeat annotation types.

      AAMI classes                    MIT-BIH heartbeats
        Non-ectopic beats (N)             Normal beats (N)
                                          Left bundle branch block beats (L)
                                          Right bundle branch block beats (R)
                                          Nodal (junctional) escape beats (j)
                                          Atrial escape beats (e)


        SVEB:
        Supra ventricular
        ectopic beats (S)                 Aberrated atrial premature beats (a)
                                          Supraventricular premature beats (S)
                                          Atrial premature beat (A)
                                          Nodal (junctional) premature beats (J)

        VEB:
        Ventricular ectopic               Ventricular escape beats (E)
        beats (V)                         Premature ventricular contraction (V)


        Fusion beats (F)                  Fusion of ventricular and normal beat (F)

        Unknown beats (Q)                 Paced beats (/)
                                          Unclassifiable beats (U)
                                          Fusion of paced and normal beats (f)

    """
    N = 0
    S = 1
    V = 2
    F = 3
    Q = 4

    @classmethod
    def from_name(cls, name):
        for label in cls:
            if name == label.name:
                return label.value
        raise ValueError("{} is no a valid label.".format(name))


def convert_heartbeat_mit_bih_to_aami(heartbeat_mit_bih):
    """Converts the heartbeat label character to the corresponding general heartbeat type according to the AAMI
    standards.

    See AAMI standards here:

    :param heartbeat_mit_bih: label character from the MIT-BIH AR database.
    :return: The corresponding heartbeat label character from the AAMI standard.
    """
    if heartbeat_mit_bih in [MitBihArrhythmiaHeartBeatTypes.N.name, MitBihArrhythmiaHeartBeatTypes.L.name,
                             MitBihArrhythmiaHeartBeatTypes.R.name, MitBihArrhythmiaHeartBeatTypes.e.name,
                             MitBihArrhythmiaHeartBeatTypes.j.name]:
        return AAMIHeartBeatTypes.N.name

    elif heartbeat_mit_bih in [MitBihArrhythmiaHeartBeatTypes.A.name, MitBihArrhythmiaHeartBeatTypes.a.name,
                               MitBihArrhythmiaHeartBeatTypes.J.name, MitBihArrhythmiaHeartBeatTypes.S.name]:
        return AAMIHeartBeatTypes.S.name

    elif heartbeat_mit_bih in [MitBihArrhythmiaHeartBeatTypes.V.name, MitBihArrhythmiaHeartBeatTypes.E.name]:
        return AAMIHeartBeatTypes.V.name

    elif heartbeat_mit_bih in [MitBihArrhythmiaHeartBeatTypes.F.name]:
        return AAMIHeartBeatTypes.F.name
    else:
        return AAMIHeartBeatTypes.Q.name


def convert_heartbeat_mit_bih_to_aami_index_class(heartbeat_mit_bih):
    """Converts the heartbeat label character to the corresponding general heartbeat type according to the AAMI
    standards index.
    The possible five indicies are:
    0 - N
    1 - S
    2 - V
    3 - F
    4 - Q
    :param heartbeat_mit_bih: heartbeat_mit_bih: label character from the MIT-BIH AR database.
    :return: he corresponding heartbeat label index from the AAMI standard.
    """
    aami_label_char = convert_heartbeat_mit_bih_to_aami(heartbeat_mit_bih)
    aami_label_ind = AAMIHeartBeatTypes.from_name(aami_label_char)
    return aami_label_ind


def convert_to_one_hot(label_ind):
    label_one_hot = [0 for _ in range(5)]
    label_one_hot[label_ind] = 1
    return label_one_hot
