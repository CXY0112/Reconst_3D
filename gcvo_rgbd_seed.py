"""Current RGB -> trusted reference RGB-D candidate, verified before submission."""
import cv2
import numpy as np


class RGBDSeed:
    def __init__(self):
        self.orb=cv2.ORB_create(nfeatures=1600);self.reference=None;self.observation=None

    def reset(self):self.reference=None;self.observation=None

    def estimate(self,reference_rgb,reference_depth,reference_k,color,k):
        self.observation=None
        if any(x is None for x in (reference_rgb,reference_depth,reference_k,color,k)):return None
        if self.reference is None:
            keypoints,descriptors=self.orb.detectAndCompute(reference_rgb,None)
            self.reference=(keypoints,descriptors)
        points,descriptors=self.reference
        current,current_desc=self.orb.detectAndCompute(color,None)
        if descriptors is None or current_desc is None or len(current_desc)<2:return None
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors,current_desc,k=2)
        matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
        # Keep a one-to-one image mapping before robust pose estimation.
        unique={}
        for match in sorted(matches,key=lambda m:m.distance):unique.setdefault(match.trainIdx,match)
        matches=list(unique.values())
        if len(matches)<20:return None
        ref=np.array([points[m.queryIdx].pt for m in matches]);image=np.array([current[m.trainIdx].pt for m in matches])
        u,v=np.rint(ref).astype(int).T;h,w=reference_depth.shape
        valid=(u>=0)&(u<w)&(v>=0)&(v<h)
        z=reference_depth[np.clip(v,0,h-1),np.clip(u,0,w-1)].astype(float)/1000
        valid&=(z>.1)&(z<5)
        ref,z,image=ref[valid],z[valid],image[valid]
        if len(z)<20:return None
        kr=np.asarray(reference_k);xyz=np.column_stack(((ref[:,0]-kr[0,2])*z/kr[0,0],(ref[:,1]-kr[1,2])*z/kr[1,1],z))
        ok,r,t,inliers=cv2.solvePnPRansac(xyz,image,np.asarray(k,dtype=float),None,
            iterationsCount=200,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<20 or len(inliers)<.3*len(xyz):return None
        selected=inliers.ravel()
        r,t=cv2.solvePnPRefineLM(xyz[selected],image[selected],np.asarray(k,dtype=float),None,r,t)
        rotation=cv2.Rodrigues(r)[0];translation=t.ravel()
        projected=cv2.projectPoints(xyz[selected],r,t,np.asarray(k,dtype=float),None)[0][:,0]
        error=float(np.median(np.linalg.norm(projected-image[selected],axis=1)))
        if not np.isfinite(error) or error>3 or not np.isfinite(translation).all():return None
        if np.mean((xyz[selected]@rotation.T+translation)[:,2]>.01)<.95:return None
        # PnP estimates current <- reference; G-CVO needs reference <- current.
        pose=np.eye(4);pose[:3,:3]=rotation.T;pose[:3,3]=-rotation.T@translation
        self.observation=(xyz[selected].copy(),image[selected].copy(),np.asarray(k,dtype=float).copy())
        return pose,dict(method='rgbd_orb_pnp',matches=len(xyz),inliers=len(selected),median_reprojection_px=error)

    def verify_depth(self,pose,current_depth):
        if self.observation is None or current_depth is None:return None
        xyz,pixels,k=self.observation;u,v=np.rint(pixels).astype(int).T;h,w=current_depth.shape
        valid=(u>=0)&(u<w)&(v>=0)&(v<h)
        z=current_depth[np.clip(v,0,h-1),np.clip(u,0,w-1)].astype(float)/1000
        valid&=(z>.1)&(z<5)
        current=np.column_stack(((pixels[:,0]-k[0,2])*z/k[0,0],(pixels[:,1]-k[1,2])*z/k[1,1],z))
        residual=np.linalg.norm(current@pose[:3,:3].T+pose[:3,3]-xyz,axis=1)
        good=valid&(residual<.04)
        if good.sum()<20 or good.mean()<.7:return None
        return dict(inliers=int(good.sum()),ratio=float(good.mean()),median_error_m=float(np.median(residual[good])))
