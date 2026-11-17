"""CPU inference adapter for the official PointDSC indoor pretrained network."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


class PointDSCMatcher:
    def __init__(self):
        import torch
        from pointdsc_impl.network import PointDSC
        torch.set_num_threads(2)
        self.torch = torch
        folder = ROOT/'models/pointdsc'
        config = json.loads((folder/'config.json').read_text())
        self.model = PointDSC(**{key:config[key] for key in ('in_dim','num_layers','num_channels','num_iterations','ratio','sigma_d','k')},
                              nms_radius=config['inlier_threshold'])
        state = torch.load(folder/'models/model_best.pkl',map_location='cpu',weights_only=True)
        # Official release carries an unused legacy scalar absent from upstream inference.
        state.pop('gamma',None)
        self.model.load_state_dict(state,strict=True)
        self.model.eval()

    def estimate(self, source, target):
        if len(source) < 40:
            raise ValueError('PointDSC 的有效几何候选对应不足 40 对')
        torch = self.torch
        corr = np.concatenate((source,target),axis=1).astype(np.float32)
        corr -= corr.mean(axis=0)
        data = {'corr_pos':torch.from_numpy(corr)[None],
                'src_keypts':torch.from_numpy(np.asarray(source,np.float32))[None],
                'tgt_keypts':torch.from_numpy(np.asarray(target,np.float32))[None],'testing':True}
        with torch.inference_mode():
            result = self.model(data)
        pose = result['final_trans'][0].numpy().astype(float)
        if not np.isfinite(pose).all():
            raise ValueError('PointDSC 返回了无效位姿')
        return pose
