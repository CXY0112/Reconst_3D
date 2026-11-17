"""Stage four provenance, one paid submission, local resume and tab lifecycle."""
from pathlib import Path
from contextlib import contextmanager
import io,json,struct,tempfile,time,unittest
from unittest.mock import patch
import cv2
import numpy as np
from scene_segmentation import ROOT,load_scene,save_segmentation
from scene_reconstruction import (load_inputs,digest,write_json,replace_base,segmentation_digest,read_mesh,run_command)
from fal_object_modeling import prepare_object,submit_object,fetch_object,read_state,validate_inputs
from fal_transport import env_file_key,http_json
from test_scene_cloud import fixture as raw_fixture


def fixture(root):
    raw=root/'setp1';raw.mkdir();raw_fixture(raw)
    from tsdf_reconstruction import sdk_frames
    frames,k,_=sdk_frames(raw);v,u=np.indices((20,24));u=u.ravel()+20;v=v.ravel()+14
    camera=np.column_stack(((u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],np.ones(len(u))))
    xyz=(camera@frames[0]['pose'][:3,:3].T+frames[0]['pose'][:3,3]).astype(np.float32)
    folder=root/'rgbd_scene/setp2';folder.mkdir(parents=True)
    np.save(folder/'processed_world.npy',xyz)
    np.savez(folder/'surfel_attributes.npz',rgb=np.full(xyz.shape,120,np.uint8),normals_world=np.tile([1.,0,0],(len(xyz),1)))
    write_json(folder/'report.json',dict(complete=True,scene_rgb_schema='colored_scene_v1',source_session_relative='../../setp1'))
    scene=load_scene(folder);scene.instance=np.where(u<32,1,2).astype(np.int32);scene.semantic=np.full(len(u),19,np.int32)
    scene.report.update(unassigned_instance_id=0,backend='rgbd_box_only')
    save_segmentation(scene,root/'rgbd_scene/setp3')
    write_json(root/'dataset.json',dict(schema_version=1,name='test',stages={}))
    return load_inputs(folder,root/'rgbd_scene/setp3')[0]


class Storage:
    def open(self,request,timeout):
        assert request.get_header('Authorization')=='Bearer upload-token'
        return io.BytesIO(json.dumps(dict(access_url='https://v3.fal.media/pointmap.npy')).encode())


def submitted(url,key,method='GET',payload=None):
    if 'storage/auth' in url:return dict(token_type='Bearer',token='upload-token')
    assert payload['pointmap_url'].startswith('https://v3.fal.media/')
    assert payload['image_url'].startswith('data:image/png;base64,')
    assert payload['mask_urls'][0].startswith('data:image/png;base64,')
    assert payload['export_textured_glb'] is True
    return dict(request_id='test-request',status_url='https://queue.fal.run/status/test-request',response_url='https://queue.fal.run/result/test-request')


class StageFourTests(unittest.TestCase):
    def test_object_preparation_has_pixel_aligned_pointmap_and_immutable_geometry(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=fixture(root);before=scene.points_world.copy()
            run,state=prepare_object(scene,1,root/'rgbd_scene/setp4/fal')
            p=np.load(run/'pointmap.npy');mask=cv2.imread(str(run/'mask.png'),0)
            valid=np.isfinite(p).all(2);v,u=np.nonzero(valid);xyz=p[valid]*[-1,-1,1];k=np.array(state['K'])
            uv=xyz[:,:2]/xyz[:,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
            np.testing.assert_allclose(uv,np.column_stack((u,v)),atol=1e-4)
            self.assertGreater(state['mask_pixels'],100);self.assertEqual(mask.shape,p.shape[:2])
            np.testing.assert_array_equal(scene.points_world,before)
            validate_inputs(run,scene)
            scene.instance[:]=2
            with self.assertRaisesRegex(ValueError,'更新'):validate_inputs(run,scene)

    def test_explicit_submit_exactly_once_and_does_not_persist_credentials(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=fixture(root);run,_=prepare_object(scene,1,root/'rgbd_scene/setp4/fal')
            calls=[]
            def transport(*a,**kw):calls.append(a[0]);return submitted(*a,**kw)
            with patch('fal_object_modeling.key_for',return_value='test-private-key'):
                state=submit_object(run,scene,transport=transport,opener=Storage())
                self.assertEqual(state['status'],'submitted')
                with self.assertRaises(ValueError):submit_object(run,scene,transport=transport,opener=Storage())
            self.assertEqual(sum(url.endswith('/3d-objects') for url in calls),1)
            self.assertNotIn('test-private-key',(run/'run_state.json').read_text(encoding='utf-8'))
            self.assertNotIn('data:image',(run/'run_state.json').read_text(encoding='utf-8'))

    def test_uncertain_submission_is_redacted_and_never_retried(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=fixture(root);run,_=prepare_object(scene,1,root/'rgbd_scene/setp4/fal')
            def transport(url,key,method,payload):
                if 'storage/auth' in url:return submitted(url,key,method,payload)
                raise TimeoutError('private-test-key timeout')
            with patch('fal_object_modeling.key_for',return_value='private-test-key'):
                with self.assertRaises(RuntimeError):submit_object(run,scene,transport=transport,opener=Storage())
                self.assertEqual(read_state(run)['status'],'submission_uncertain')
                self.assertNotIn('private-test-key',(run/'run_state.json').read_text(encoding='utf-8'))
                with self.assertRaises(ValueError):submit_object(run,scene,transport=transport,opener=Storage())

    def test_fetch_download_and_offline_local_resume_without_api_key(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);scene=fixture(root);run,_=prepare_object(scene,1,root/'rgbd_scene/setp4/fal')
            with patch('fal_object_modeling.key_for',return_value='private-test-key'):
                submit_object(run,scene,transport=submitted,opener=Storage())
                def transport(url,key):
                    return dict(status='COMPLETED') if '/status/' in url else dict(model_glb=dict(url='https://v3.fal.media/generated.glb'),metadata=[])
                def download(url,path):path.write_bytes(b'glTF'+struct.pack('<II',2,12))
                self.assertEqual(fetch_object(run,transport=transport,downloader=download)['status'],'downloaded')
            with patch('fal_object_modeling.key_for',side_effect=AssertionError('must not read key')):
                self.assertEqual(fetch_object(run)['status'],'downloaded')
            self.assertFalse(read_state(run)['merged_into_scene'])

    def test_base_transaction_failure_preserves_previous_and_foreign_folder_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            root=Path(tmp);target=root/'base';target.mkdir()
            write_json(target/'report.json',dict(schema='rgbd_scene_base_v1',complete=True));(target/'keep').write_text('old')
            with self.assertRaises(ValueError):
                with replace_base(target):pass
            self.assertEqual((target/'keep').read_text(),'old')
            foreign=root/'foreign';foreign.mkdir();write_json(foreign/'report.json',dict(schema='other'))
            with self.assertRaises(ValueError):
                with replace_base(foreign):pass

    def test_env_literal_parser_and_api_key_host_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            p=Path(tmp)/'.env';p.write_text('export FAL_KEY="literal-test-key"\nIGNORED=anything',encoding='utf-8')
            self.assertEqual(env_file_key(p),'literal-test-key')
        with self.assertRaises(ValueError):http_json('https://untrusted.example/status','test-key')

    def test_glb_texture_preview_keeps_node_transform_and_original_bytes(self):
        from runtime_paths import geometry_python
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            source=Path(tmp)/'textured.glb'
            code=(f'import sys;sys.path.insert(0,{str(ROOT)!r});'
                'from runtime_paths import activate_geometry_packages;activate_geometry_packages();'
                'import trimesh,numpy as np;from PIL import Image;'
                'm=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False);'
                'm.visual=trimesh.visual.texture.TextureVisuals(uv=np.array([[0,0],[1,0],[0,1]]),image=Image.new("RGB",(4,4),(100,150,200)));'
                's=trimesh.Scene();t=np.eye(4);t[:3,3]=[2,3,4];s.add_geometry(m,transform=t);'
                f's.export({str(source)!r});')
            run_command([geometry_python(),'-c',code])
            before=digest(source);mesh,colors=read_mesh(source)
            np.testing.assert_allclose(mesh[0].min(0),[2,3,4]);np.testing.assert_allclose(mesh[0].max(0),[3,4,4])
            np.testing.assert_allclose(colors,np.tile([100/255,150/255,200/255],(3,1)),atol=.01)
            self.assertEqual(before,digest(source))

    def test_workbench_new_stage_stale_propagation(self):
        import tkinter as tk
        from scene_workbench import SceneWorkbench
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            folder=Path(tmp);fixture(folder);root=tk.Tk();root.withdraw();app=SceneWorkbench(root);app.dataset=folder
            try:
                self.assertEqual(len(app.tabs.tabs()),5)
                app.stage_saved(4,folder/'rgbd_scene/setp4')
                app.stage_saved(3,folder/'rgbd_scene/setp3')
                manifest=json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
                self.assertTrue(manifest['branches']['rgbd_scene']['stages']['4']['stale'])
                self.assertEqual(manifest['branches']['rgbd_scene']['workflow_version'],2)
                app.stage_saved(1,folder/'setp1')
                manifest=json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
                self.assertTrue(manifest['branches']['rgbd_scene']['stages']['3']['stale'])
            finally:app.close()


if __name__=='__main__':unittest.main(verbosity=2)
