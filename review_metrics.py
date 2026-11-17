"""Human-readable saved metrics; missing historical measurements stay unknown."""
import math
from saved_sources import metadata


def describe(info, raw_count=None):
    pre=metadata(info['preprocessed']) if info.get('preprocessed') else {}
    geom=metadata(info['geometry']) if info.get('geometry') else {}
    color=info['report'];final=geom.get('final',{})
    coverage=final.get('input_coverage',geom.get('metrics',{}).get('input_coverage',{}))
    rgb=color.get('coverage',{})
    def number(value):return f'{value:,}' if isinstance(value,(int,float)) else '未记录'
    def percent(value):return f'{value:.1%}' if isinstance(value,(int,float)) else '未记录'
    error=final.get('input_to_mesh_mm_p95')
    error_text=f'{error:.2f} mm' if isinstance(error,(int,float)) else '未记录'
    mesh=color.get('final',final)
    method=geom.get('method') or ('poisson' if geom.get('pipeline','').startswith('poisson') else None)
    method={'poisson':'Poisson（平面优化）','nksr':'NKSR','noksr':'NoKSR（历史结果）','tsdf':'SDK深度TSDF'}.get(method,'未记录')
    original_count=raw_count if raw_count is not None else pre.get('input_points')
    processed_count=pre.get('output_points')
    reduction='减少比例未记录'
    if (isinstance(original_count,(int,float)) and math.isfinite(original_count) and original_count>0
            and isinstance(processed_count,(int,float)) and math.isfinite(processed_count) and processed_count>=0):
        fraction=1-processed_count/original_count
        reduction=f'减少 {fraction:.1%}' if fraction>=0 else f'增加 {-fraction:.1%}'
    processed_text='预处理已跳过' if pre.get('skipped') else f'预处理后 {number(processed_count)} 点（{reduction}）'
    return '\n'.join([
        f"点云：原始融合 {number(original_count)} 点；{processed_text}；拍摄 {number(pre.get('captures',geom.get('captures')))} 次；原始 RGB {len(info.get('frames',[]))} 张。",
        f"几何：{method}；顶点 {number(mesh.get('vertices'))}；三角面 {number(mesh.get('triangles'))}。",
        f"几何贴合：10 mm 内覆盖率 {percent(coverage.get('10',coverage.get('10.0')))}；点到网格距离 P95 {error_text}。",
        f"颜色：RGB 实测顶点覆盖率 {percent(rgb.get('vertex_fraction'))}；插值补色 {number(rgb.get('interpolated_vertices'))} 顶点；含补色顶点覆盖率 {percent(rgb.get('display_vertex_fraction'))}；完整着色面积覆盖率 {percent(rgb.get('display_fully_colored_area_fraction'))}。"
    ])
