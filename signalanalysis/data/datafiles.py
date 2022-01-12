from pkg_resources import resource_filename

__all__ = [
    "LOBACHEVSKY",
    "EGM_BIPOLAR",
    "EGM_UNIPOLAR",
    "PTB_DATABASE",
    "PTB_100",
    "PTB_500",
    "ELECTRODES",
]

LOBACHEVSKY = resource_filename(__name__,
                                "lobachevsky/3")

EGM_BIPOLAR = resource_filename(__name__,
                                "egm/egm_bipolar.csv")
EGM_UNIPOLAR = resource_filename(__name__,
                                "egm/egm_unipolar.csv")

PTB_DATABASE = resource_filename(__name__,
                                 "ptb-xl/ptbxl_database.csv"
)
PTB_100 = resource_filename(__name__,
                            "ptb-xl/records100/00000/00001_lr")
PTB_500 = resource_filename(__name__,
                            "ptb-xl/records500/00000/00001_hr")

ELECTRODES = resource_filename(__name__,
                               "12LeadElectrodes.dat")