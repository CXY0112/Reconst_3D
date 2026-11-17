import unittest
import numpy as np
from color_completion import complete_colors
from appearance_pipeline import observe


def grid(n=9,spacing=.005):
    points=np.array([[x*spacing,y*spacing,1.] for y in range(n) for x in range(n)])
    faces=[]
    for y in range(n-1):
        for x in range(n-1):
            a=y*n+x;faces.extend([[a,a+1,a+n],[a+1,a+n+1,a+n]])
    return points,np.array(faces),np.tile([0,0,1.],(n*n,1))


class CompletionTests(unittest.TestCase):
    def test_small_interior_gap_fills_without_touching_observations(self):
        points,faces,normals=grid();covered=np.ones(len(points),bool);covered[40]=False
        colors=np.tile([.1,.2,.3],(len(points),1));colors[40]=.65
        original=points.copy();out,filled,labels,stats=complete_colors(points,faces,normals,colors,covered)
        self.assertEqual(filled.sum(),1);self.assertTrue(filled[40]);self.assertFalse(np.any(filled & covered))
        np.testing.assert_allclose(out[40],[.1,.2,.3],atol=1e-12)
        np.testing.assert_array_equal(out[covered],colors[covered]);np.testing.assert_array_equal(points,original)
        self.assertFalse(stats['semantic_objects'])

    def test_open_boundary_large_missing_region_and_geodesic_limit_stay_unknown(self):
        points,faces,normals=grid();colors=np.full(points.shape,.3);covered=np.ones(len(points),bool)
        covered[0]=False
        _,filled,_,_=complete_colors(points,faces,normals,colors,covered);self.assertFalse(filled.any())
        covered[:]=False;covered[:9]=True
        _,filled,_,_=complete_colors(points,faces,normals,colors,covered);self.assertFalse(filled.any())
        covered[:]=True;covered[40]=False
        _,filled,_,_=complete_colors(points,faces,normals,colors,covered,max_distance=.001)
        self.assertFalse(filled.any())

    def test_disconnected_nearby_object_and_sharp_normal_partition_do_not_leak(self):
        points,faces,normals=grid();count=len(points)
        points=np.r_[points,points+[0,0,.001]];faces=np.r_[faces,faces+count];normals=np.r_[normals,-normals]
        covered=np.r_[np.ones(count,bool),np.zeros(count,bool)];covered[40]=False
        colors=np.full(points.shape,.65);colors[:count]=[.05,.1,.2];colors[40]=.65
        out,filled,labels,_=complete_colors(points,faces,normals,colors,covered)
        self.assertTrue(filled[40]);self.assertFalse(filled[count:].any())
        self.assertNotEqual(labels[40],labels[count+40])
        normals[:count]=[0,0,1];normals[40]=[0,0,-1]
        _,filled,_,_=complete_colors(points,faces,normals,colors,covered)
        self.assertFalse(filled[40])

    def test_texture_boundary_not_smeared_across_missing_color(self):
        points,faces,normals=grid();covered=np.ones(len(points),bool);covered[40]=False
        colors=np.zeros(points.shape);colors[points[:,0]>.02]=1
        _,filled,_,_=complete_colors(points,faces,normals,colors,covered)
        self.assertFalse(filled[40])

    def test_recovered_depth_is_lower_weight_and_still_rejects_foreground_occlusion(self):
        k=np.array([[50,0,32],[0,50,32],[0,0,1.]])
        vertices=np.array([[0,0,1.],[.1,0,1.023],[.2,0,1.06]])
        normals=np.tile([0,0,1.],(3,1));image=np.full((64,64,3),128.);depth=np.ones((64,64))
        strict=np.zeros(3,bool)
        ids,_,weights=observe(vertices,normals,{'pose':np.eye(4)},k,image,depth,enhanced=True,strict_seen=strict)
        np.testing.assert_array_equal(ids,[0,1]);np.testing.assert_array_equal(strict,[True,False,False])
        self.assertLess(weights[1],weights[0]*.4)
