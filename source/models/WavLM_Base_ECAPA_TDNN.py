from s3prl.hub import wavlm_base
from .s3prl_ECAPA_TDNN import s3prl_ECAPA_TDNN


class WavLM_Base_ECAPA_TDNN(s3prl_ECAPA_TDNN):
    def hub_function(self):
        return wavlm_base()
