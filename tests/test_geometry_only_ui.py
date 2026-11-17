"""Geometry-only contracts, local object browsing and Beta modal behavior."""
import tempfile,unittest,tkinter as tk
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scene_segmentation import ROOT,region_colors
from scene_reconstruction import replace_base,write_json
from test_scene_reconstruction import fixture


class GeometryOnlyUI(unittest.TestCase):
    def test_transaction_accepts_geometry_and_rejects_missing_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            target=Path(tmp)/'base'
            with replace_base(target) as pending:
                (pending/'geometry').mkdir();(pending/'geometry/mesh.ply').write_bytes(b'geometry fixture')
                write_json(pending/'report.json',dict(schema='rgbd_scene_base_v1',complete=True,geometry_only=True,model_relative='geometry/mesh.ply'))
            with self.assertRaises(ValueError):
                with replace_base(target) as pending:
                    write_json(pending/'report.json',dict(schema='rgbd_scene_base_v1',complete=True,model_relative='../outside.ply'))
            self.assertEqual((target/'geometry/mesh.ply').read_bytes(),b'geometry fixture')
            self.assertFalse((target/'color').exists())

    def test_region_preview_preserves_rgb_and_geometry(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            scene=fixture(Path(tmp));points=scene.points.copy();rgb=scene.rgb.copy()
            colors=region_colors(scene,selected=[1],pending=[0],assigned=False)
            self.assertTrue((colors[scene.instance!=1]==160).all())
            np.testing.assert_array_equal(colors[0],[246,190,93])
            np.testing.assert_array_equal(scene.points,points);np.testing.assert_array_equal(scene.rgb,rgb)

    def test_beta_file_picker_selection_filter_and_overwrite_cancel(self):
        from beta_theme import install
        from ui_dialogs import FilePicker,themed
        root=tk.Tk();root.withdraw();install(root)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
                folder=Path(tmp);(folder/'sub').mkdir();(folder/'old.ply').write_text('keep');(folder/'hidden.txt').write_text('keep')
                d=FilePicker(root,'save',initialdir=folder,initialfile='new',defaultextension='.ply',filetypes=[('PLY','*.ply')])
                self.assertTrue(themed(d.window))
                self.assertEqual({p.name for p in d.entries.values()},{'sub','old.ply'})
                d.filename.set('old.ply')
                with patch('ui_dialogs.message',return_value=False):d.accept()
                self.assertIsNone(d.result);self.assertEqual((folder/'old.ply').read_text(),'keep')
                d.filename.set('new');d.accept();self.assertEqual(Path(d.result),folder/'new.ply')
                self.assertFalse((folder/'new.ply').exists())
                d=FilePicker(root,'directory',initialdir=folder)
                ident=next(i for i,p in d.entries.items() if p.name=='sub');d.tree.selection_set(ident);d.accept()
                self.assertEqual(Path(d.result),folder/'sub')
        finally:root.destroy()

    def test_regular_dialog_dispatch_remains_native(self):
        from ui_dialogs import simpledialog,filedialog
        root=tk.Tk();root.withdraw()
        try:
            with patch('ui_dialogs.native_simple.askstring',return_value='dataset') as native:
                self.assertEqual(simpledialog.askstring('Name','Name',parent=root),'dataset');native.assert_called_once()
            with patch('ui_dialogs.native_files.askdirectory',return_value='folder') as native:
                self.assertEqual(filedialog.askdirectory(parent=root),'folder');native.assert_called_once()
        finally:root.destroy()


if __name__=='__main__':unittest.main()
