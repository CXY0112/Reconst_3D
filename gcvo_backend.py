"""Selectable G-CVO geometry tracking with the existing iSAM2/submap lifecycle."""
from functools import partial
import json
import cv2
import numpy as np
from bufferx_backend import BufferXBackend
from bufferx_backend import validate_registration
from gcvo_matcher import GCVOMatcher
from gcvo_tracking import GCVOTracking


class GCVOConfidenceDeferred(ValueError):pass


class GCVOBackend(BufferXBackend):
    def __init__(self,folder,optimizer='isam2',device='cpu',matcher_factory=None,client_factory=None,mac_factory=None):
        self.backend_name='gcvo_'+device
        self.frontend_name='G-CVO '+device.upper()
        super().__init__(folder,optimizer=optimizer,matcher_factory=matcher_factory or partial(GCVOMatcher,device=device),client_factory=client_factory)
        try:self.tracking=GCVOTracking(self.settings)
        except Exception:
            self.close();raise
        (self.folder/'gcvo_tracking_config.json').write_text(json.dumps(self.tracking.config,indent=2),encoding='utf-8')
        self.reference_preview=None;self._reference_revision=0;self._frame_rgb=None
        self.mac=None;self._mac_used=False
        if self.settings.get('mac',{}).get('enabled',False):
            from mac_recovery import MACRecovery
            try:self.mac=(mac_factory or MACRecovery)(self.folder,self.settings)
            except Exception:
                self.close();raise

    def process(self,color,depth,k,stamp,geometry=None):
        self._frame_rgb=color;self._mac_used=False;self.recovery_attempts=[]
        try:return super().process(color,depth,k,stamp,geometry)
        finally:self._frame_rgb=None

    def resume_tracking(self,stamp):
        self._recovery_pose=None
        self.tracking.resume(stamp)
        if hasattr(self.matcher,'reset_prediction'):self.matcher.reset_prediction()
        if self.mac is not None:self.mac.reset()

    def _estimate(self,source,target,stamp):
        # Prefer G-CVO refinement, then permit independently verified recovery.
        # Both routes retain geometry, confidence and native acceptance gates.
        if self.tracking.state=='RECOVERY' and hasattr(self.matcher,'recovery_seed'):
            seed=self.matcher.recovery_seed()
            if seed is not None:
                pose,diagnostics=seed
                try:
                    result=self._recovery_estimate(source,target,pose,diagnostics)
                    return {**result,'initialization':diagnostics}
                except ValueError as exc:self.recovery_attempts.append(dict(method='RGB-D ORB/PnP',error=str(exc)))
        if self.mac is not None and self.tracking.state=='RECOVERY':
            seed=self.mac.poll(self.reference_node,source,target,stamp)
            if seed is not None:
                try:
                    result=self._recovery_estimate(source,target,seed,dict(method='mac'));self._mac_used=True
                    return result
                except ValueError as exc:
                    self.recovery_attempts.append(dict(method='MAC + ICP',error=str(exc)))
                    self.mac.reject_seed('Current-frame recovery verification rejected MAC: '+str(exc))
        return self.matcher.estimate(source,target)

    def _recovery_estimate(self,source,target,pose,diagnostics):
        try:
            result=self.matcher.estimate(source,target,initial=pose)
            validate_registration(source,target,result,self.settings,self.frontend_name)
            return result
        except ValueError as exc:
            if not hasattr(self.matcher,'estimate_fallback'):raise
            result=self.matcher.estimate_fallback(source,target,pose,diagnostics)
            result['gcvo_rejection']=str(exc)
            return result

    def _before_submit(self,details,metrics,stamp,initial=False):
        if details.get('independent_recovery'):
            pose=np.asarray(details['pose']);previous=getattr(self,'_recovery_pose',None)
            if previous is not None:
                delta=np.linalg.inv(previous)@pose
                if np.linalg.norm(delta[:3,3])>.08 or np.trace(delta[:3,:3])<1+2*np.cos(np.deg2rad(8)):
                    self.tracking.good_count=0
            self._recovery_pose=pose.copy()
        else:self._recovery_pose=None
        if not self.tracking.assess(details,metrics,stamp,initial):
            t=self.tracking
            reason=(f'RECOVERY：暂停位姿提交，请回到最后可信视角；连续确认 {t.good_count}/{t.config["recovery_good_frames"]}'
                    if t.state=='RECOVERY' else '置信度偏低，本帧暂停提交')
            raise GCVOConfidenceDeferred(reason)

    def _registration_failed(self,exc,details,metrics,stamp):
        if not isinstance(exc,GCVOConfidenceDeferred):
            self._recovery_pose=None
            self.tracking.failure(stamp,str(exc))
            self.tracking.retried=bool(getattr(exc,'diagnostics',details).get('retried'))

    def _after_submit(self,status,stamp):
        self._recovery_pose=None
        self.tracking.submitted(status,stamp)
        if self.mac is not None and status>0 and self.tracking.state=='NORMAL':self.mac.reset()
        if status==2 and self._frame_rgb is not None:
            color=np.asarray(self._frame_rgb)
            if color.ndim==3 and color.shape[2]==3 and color.dtype==np.uint8 and color.size:
                h,w=color.shape[:2];scale=min(240/w,135/h,1.)
                image=cv2.resize(color,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
                image.setflags(write=False);self._reference_revision+=1
                self.reference_preview=dict(revision=self._reference_revision,node=self.reference_node,stamp=float(stamp),image=image)

    def _record(self,stamp):
        self.report['tracking']=self.tracking.snapshot()
        if self.recovery_attempts:self.report['recovery_attempts']=self.recovery_attempts
        if self.report.get('frontend',{}).get('initialization',{}).get('method')=='rgbd_orb_pnp':
            self.report['tracking']['recovery_method']='rgbd_seed_then_gcvo'
        self.report['tracking']['mac_enabled']=self.mac is not None
        if self.mac is not None:
            self.report['mac']={**self.mac.snapshot(),'used_seed':self._mac_used}
            self.report['tracking']['mac_status']=self.report['mac'].get('status','idle')
            if self._mac_used:self.report['tracking']['recovery_method']='mac_seed_then_gcvo'
        frontend=self.report.get('frontend',{})
        if frontend.get('independent_recovery'):
            self.report['method']=frontend['registration_method']
            self.report['tracking']['recovery_method']=frontend['registration_method']
        if self.reference_preview is not None:
            self.report['tracking']['reference_view']={k:v for k,v in self.reference_preview.items() if k!='image'}
        while self.tracking.events:
            event=self.tracking.events.pop(0)
            with (self.folder/'gcvo_tracking_events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(event,ensure_ascii=False)+'\n')
        super()._record(stamp)

    def close(self):
        if getattr(self,'mac',None) is not None:self.mac.close()
        super().close()
