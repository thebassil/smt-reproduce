# Disable fp16 autocast: add-pooling sums hundreds of node embeddings,
# overflowing fp16 range (max 65504) and producing NaN scores.
import torch.cuda.amp as _amp
_orig = _amp.autocast
class _NoAutocast(_orig):
    def __init__(self, *a, **kw):
        kw['enabled'] = False
        super().__init__(*a, **kw)
_amp.autocast = _NoAutocast
