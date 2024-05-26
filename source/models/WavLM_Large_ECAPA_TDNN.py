from s3prl.hub import wavlm_large
from .s3prl_ECAPA_TDNN import s3prl_ECAPA_TDNN


class WavLM_Large_ECAPA_TDNN(s3prl_ECAPA_TDNN):
    def hub_function(self):
        return wavlm_large()
