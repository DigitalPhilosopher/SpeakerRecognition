from s3prl.hub import wavlm_base
from .s3prl_ECAPA_TDNN import s3prl_ECAPA_TDNN
from s3prl.nn import S3PRLUpstream


class WavLM_Base_ECAPA_TDNN(s3prl_ECAPA_TDNN):
    def hub_function(self):
        # return wavlm_base()
        model = S3PRLUpstream("wavlm_base")
        model.upstream.model.feature_grad_mult = 1.0
        return model