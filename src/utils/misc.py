import torch
import numpy as np
from openbabel import openbabel
from rdkit import RDLogger


def disable_obabel_and_rdkit_logging():
    RDLogger.DisableLog("rdApp.*")
    openbabel.obErrorLog.SetOutputLevel(0)
    openbabel.obErrorLog.StopLogging()
    message_handler = openbabel.OBMessageHandler()
    message_handler.SetOutputLevel(0)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output