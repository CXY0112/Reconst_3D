"""Prepare selected instance RGB/mask/SDK pointmap; one explicit paid job per run."""
from pathlib import Path
from datetime import datetime
import os,time,uuid,urllib.request,urllib.parse
import cv2
import numpy as np
from scipy.spatial import cKDTree
from preprocess import _check
from scene_reconstruction import write_json,digest,segmentation_digest
from scene_segmentation import fingerprint
from rgbd_instance_segmentation import source_frames,project_visible,observation_fingerprint
from fal_transport import credentials,data_uri,http_json,NoRedirect,download_glb,safe_error
import json

MODEL='fal-ai/sam-3/3d-objects'


def key_for(env_file=None):
    if env_file:return credentials(env_file)
    explicit=os.environ.get('FAL_ENV_FILE')
    if explicit:return credentials(explicit)
    # Existing pilot configuration can be reused locally; never copy it to results.
    legacy=Path(__file__).resolve().parents[1]/'experiments/generative_completion/.env'
    return credentials(legacy if legacy.is_file() else None)


def prepare_object(scene,ident,parent,cancel=None,capture=None):
    if scene.instance is None or ident==scene.report.get('unassigned_instance_id'):raise ValueError('请先保存并选择第四阶段物体')
    selected=np.flatnonzero(scene.instance==int(ident))
    if scene.report.get('_segmentation_file') and digest(scene.report['_segmentation_file'])!=scene.report['_segmentation_file_sha256']:
        raise ValueError('第四阶段标签已更新，请重新读取后准备物体')
    if len(selected)<30:raise ValueError('物体点数过少，至少需要30个点')
    parent=Path(parent).resolve()
    if parent==scene.source or parent in scene.source.parents or scene.source in parent.parents:raise ValueError('FAL输出不能写入源点云目录')
    frames,k=source_frames(scene);raw_hash=observation_fingerprint(scene)
    if scene.report.get('rgbd_observation_fingerprint',raw_hash)!=raw_hash:raise ValueError('原始观测已变化，请重新分割并保存')
    scores=[]
    for frame in frames:
        _check(cancel)
        if capture is not None and frame['capture']!=capture:continue
        ids,_,_=project_visible(scene.points_world[selected],frame,k)
        scores.append((len(ids),frame))
    if not scores:raise ValueError('所选照片不可用')
    visible,frame=max(scores,key=lambda row:row[0])
    if visible<30:raise ValueError('没有足够的物体可见点，请补充拍摄或分割')
    rgb=cv2.imread(str(frame['rgb']),cv2.IMREAD_COLOR);depth=cv2.imread(str(frame['depth']),cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or rgb.shape[:2]!=depth.shape:raise ValueError('原始RGB/SDK深度网格不一致')
    v,u=np.indices(depth.shape);z=depth.astype(np.float32)*.001
    pointmap=np.stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z),axis=-1).astype(np.float32)
    valid=(depth>=150)&(depth<=4000);pointmap[~valid]=np.nan
    camera=pointmap[valid];world=camera@frame['pose'][:3,:3].T+frame['pose'][:3,3]
    distance,index=cKDTree(scene.points_world).query(world,k=1,workers=2)
    support=(distance<=.030)&(scene.instance[index]==int(ident))
    mask=np.zeros(depth.shape,np.uint8);mask[valid]=support.astype(np.uint8)*255
    # Close only tiny pixel gaps; no box-wide background fill or invented depth.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));mask[~valid]=0
    if (mask>0).sum()<100:raise ValueError('可用物体掩码太小，请完善第四阶段分割')
    run=parent/f'object_{int(ident):06d}'/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    cv2.imwrite(str(run/'rgb.png'),rgb);cv2.imwrite(str(run/'mask.png'),mask)
    np.save(run/'pointmap.npy',pointmap*np.array([-1.,-1.,1.],np.float32))
    np.save(run/'selected_indices.npy',selected)
    overlay=rgb.copy();m=mask>0;overlay[m]=(overlay[m]*.4+np.array([60,210,80])*.6).astype(np.uint8)
    cv2.imwrite(str(run/'input_preview.png'),overlay)
    if fingerprint(scene.source)!=scene.report['source_fingerprint'] or observation_fingerprint(scene)!=raw_hash:
        raise ValueError('准备输入时源数据发生变化，未提交云端')
    report=dict(schema='fal_object_run_v1',stage=4,status='prepared',object_id=int(ident),model=MODEL,
        source_relative=os.path.relpath(scene.source,run),source_fingerprint=scene.report['source_fingerprint'],
        segmentation_fingerprint=segmentation_digest(scene),rgbd_observation_fingerprint=raw_hash,
        capture=int(frame['capture']),stamp=frame['depth'].stem,visible_scene_points=visible,
        mask_pixels=int(m.sum()),pointmap_convention='pytorch3d -X,-Y,+Z',pointmap_source='original SDK RGB-grid depth, metres',
        K=k.tolist(),world_from_sdk=frame['pose'].tolist(),export_textured_glb=True,
        input_sha256={name:digest(run/name) for name in ('rgb.png','mask.png','pointmap.npy','selected_indices.npy')},
        automatic_resubmission=False,merged_into_scene=False,alignment_status='not verified; independent object model',
        object_name=next((o['name'] for o in scene.objects() if o['id']==int(ident)),f'物体 {ident}'))
    write_json(run/'run_state.json',report)
    write_json(run.parent/'latest.json',dict(run=run.name))
    return run,report


def read_state(run):return json.loads((Path(run)/'run_state.json').read_text(encoding='utf-8'))


def validate_inputs(run,scene=None):
    run=Path(run);state=read_state(run)
    if state.get('schema')!='fal_object_run_v1':raise ValueError('不是本阶段的FAL任务')
    for name,expected in state['input_sha256'].items():
        if name not in ('rgb.png','mask.png','pointmap.npy','selected_indices.npy') or digest(run/name)!=expected:raise ValueError('FAL输入文件已改变')
    if scene is not None:
        if scene.report.get('_segmentation_file') and digest(scene.report['_segmentation_file'])!=scene.report['_segmentation_file_sha256']:
            raise ValueError('第四阶段标签已更新，请重新读取和准备输入')
        if segmentation_digest(scene)!=state['segmentation_fingerprint'] or fingerprint(scene.source)!=state['source_fingerprint']:
            raise ValueError('物体分割/来源已更新，请重新准备输入')
        if observation_fingerprint(scene)!=state['rgbd_observation_fingerprint']:raise ValueError('原始RGB-D已更新，请重新准备输入')
    return state


def submit_object(run,scene,env_file=None,cancel=None,transport=http_json,opener=None):
    run=Path(run);state=validate_inputs(run,scene)
    if state['status']!='prepared':raise ValueError('该任务已经提交或状态不确定，请刷新已有任务；如需重做请重新准备输入')
    _check(cancel);key=key_for(env_file)
    # This exclusive marker is persisted before any inference POST. A timeout
    # must not turn a second button click into a duplicate charged generation.
    with (run/'submission.lock').open('x',encoding='utf-8') as stream:stream.write('one inference POST only\n')
    state['status']='uploading';write_json(run/'run_state.json',state);posted=False
    try:
        token=transport('https://rest.fal.ai/storage/auth/token?storage_type=fal-cdn-v3',key,'POST',{})
        request=urllib.request.Request('https://v3.fal.media/files/upload',data=(run/'pointmap.npy').read_bytes(),method='POST',
            headers={'Content-Type':'application/octet-stream','X-Fal-File-Name':'pointmap.npy',
                     'Authorization':token['token_type']+' '+token['token']})
        with (opener or urllib.request.build_opener(NoRedirect())).open(request,timeout=60) as response:
            info=json.loads(response.read(2**20))
        public=info['access_url'];parsed=urllib.parse.urlparse(public)
        if parsed.scheme!='https' or parsed.username or not (parsed.hostname or '').endswith(('.fal.media','.fal.ai')):
            raise ValueError('未知的FAL点图存储地址')
        _check(cancel)
        body=dict(image_url=data_uri(run/'rgb.png','image/png'),mask_urls=[data_uri(run/'mask.png','image/png')],
            pointmap_url=public,seed=42,export_textured_glb=bool(state['export_textured_glb']))
        state['status']='submission_started';write_json(run/'run_state.json',state);posted=True
        response=transport('https://queue.fal.run/'+MODEL,key,'POST',body)
        for field in ('request_id','status_url','response_url'):
            if not response.get(field):raise ValueError('FAL返回缺少任务编号或结果地址，不会自动重复提交')
        state.update(status='submitted',**{k:response[k] for k in ('request_id','status_url','response_url')})
        write_json(run/'run_state.json',state);return state
    except Exception as exc:
        state.update(status='submission_uncertain' if posted else 'preparation_failed',error=safe_error(str(exc),key))
        write_json(run/'run_state.json',state)
        raise RuntimeError(safe_error(str(exc),key)) from None


def fetch_object(run,env_file=None,cancel=None,transport=http_json,downloader=download_glb):
    run=Path(run);state=read_state(run)
    if state.get('status')=='downloaded' and (run/'generated.glb').is_file():
        from fal_transport import validate_glb
        validate_glb(run/'generated.glb')
        if digest(run/'generated.glb')!=state['sha256']:raise ValueError('本地FAL模型文件已改变')
        return state
    if not state.get('request_id'):raise ValueError('没有已确认的任务编号；请检查FAL控制台，不会自动重新提交')
    _check(cancel);key=key_for(env_file)
    status=transport(state['status_url'],key);state['queue_status']=status.get('status','UNKNOWN')
    write_json(run/'run_state.json',state)
    if state['queue_status']!='COMPLETED':return state
    if status.get('error'):raise RuntimeError('FAL任务失败：'+str(safe_error(status['error'],key)))
    _check(cancel);result=transport(state['response_url'],key)
    file=result.get('model_glb') or next(iter(result.get('individual_glbs',[])),None)
    if not isinstance(file,dict) or not file.get('url'):raise RuntimeError('FAL没有返回GLB网格；不会把高斯PLY当作三角网格')
    write_json(run/'result_metadata.json',safe_error(result,key))
    destination=run/'generated.glb'
    if not destination.is_file():downloader(file['url'],destination)
    from fal_transport import validate_glb
    validate_glb(destination)
    state.update(status='downloaded',model_relative='generated.glb',sha256=digest(destination))
    write_json(run/'run_state.json',state);return state
