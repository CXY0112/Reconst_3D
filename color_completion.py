"""Small color-hole interpolation on segmented mesh adjacency, never on spatial KNN."""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve


def complete_colors(vertices, triangles, normals, colors, observed, max_distance=.04,
                    max_area=.0025, max_vertices=200, object_labels=None):
    vertices=np.asarray(vertices);triangles=np.asarray(triangles);normals=np.asarray(normals)
    colors=np.asarray(colors,float);observed=np.asarray(observed,bool)
    count=len(vertices)
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    edges,multiplicity=np.unique(edges,axis=0,return_counts=True)
    boundary=np.zeros(count,bool);boundary[edges[multiplicity!=2].ravel()]=True
    a,b=edges.T;delta=vertices[a]-vertices[b];length=np.linalg.norm(delta,axis=1)
    normal=normals/np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
    # Oriented normals: opposite sides must not share color, even if very close.
    good=(np.sum(normal[a]*normal[b],axis=1)>=np.cos(np.deg2rad(40))) & (length>1e-9) & (length<=.05)
    if object_labels is not None:
        object_labels=np.asarray(object_labels)
        if object_labels.shape!=(count,) or object_labels.dtype.kind not in 'iu':raise ValueError('Invalid object labels')
        good &= (object_labels[a]>=0) & (object_labels[a]==object_labels[b])
    a,b,length=a[good],b[good],length[good]
    graph=coo_matrix((np.r_[length,length],(np.r_[a,b],np.r_[b,a])),shape=(count,count)).tocsr()
    _,regions=connected_components(graph,directed=False)
    unknown=np.flatnonzero(~observed)
    filled=np.zeros(count,bool);result=colors.copy()
    stats={'method':'normal-gated mesh components + bounded harmonic color interpolation',
           'semantic_objects':False,'object_mask_constraint':object_labels is not None,
           'surface_regions':int(regions.max()+1) if count else 0,
           'max_geodesic_mm':max_distance*1000,'max_hole_area_m2':max_area,'max_hole_vertices':max_vertices,
           'filled_components':0,'rejected_components':0}
    if not observed.any() or not len(unknown):return result,filled,regions,stats
    distances=dijkstra(graph,directed=False,indices=np.flatnonzero(observed),min_only=True,limit=max_distance)
    _,holes=connected_components(graph[unknown][:,unknown],directed=False)
    area=np.linalg.norm(np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],
                                vertices[triangles[:,2]]-vertices[triangles[:,0]]),axis=1)/2
    vertex_area=np.bincount(triangles.ravel(),weights=np.repeat(area/3,3),minlength=count)
    weights=graph.copy();weights.data=1/np.maximum(weights.data,1e-6)
    linear=np.where(colors<=.04045,colors/12.92,((colors+.055)/1.055)**2.4)
    for group in range(int(holes.max()+1)):
        ids=unknown[holes==group]
        if len(ids)>max_vertices or boundary[ids].any() or not np.isfinite(distances[ids]).all() or vertex_area[ids].sum()>max_area:
            stats['rejected_components']+=1;continue
        neighbors=np.unique(graph[ids].indices);anchors=neighbors[observed[neighbors]]
        if len(anchors)<3:
            stats['rejected_components']+=1;continue
        # Small smooth color gaps only. Do not synthesize across strong texture edges.
        spread=np.sqrt(np.mean((colors[anchors]-np.median(colors[anchors],axis=0))**2,axis=1))
        if np.percentile(spread,90)>.15:
            stats['rejected_components']+=1;continue
        adjacency=weights[ids]
        laplacian=diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency[:,ids]
        values=spsolve(laplacian.tocsc(),adjacency[:,anchors]@linear[anchors])
        values=np.asarray(values).reshape(-1,3)
        if not np.isfinite(values).all():
            stats['rejected_components']+=1;continue
        values=np.clip(values,0,1)
        result[ids]=np.where(values<=.0031308,12.92*values,1.055*values**(1/2.4)-.055)
        filled[ids]=True;stats['filled_components']+=1
    stats['filled_vertices']=int(filled.sum())
    return np.clip(result,0,1),filled,regions,stats
