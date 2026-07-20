"""Faithful author-code reconstruction of paper Figures 5--8."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantum_geometry import (berry_curvature, coherent_states, expectation_values,
                              matrix_laplacian, uncertainty)


CONFIG = {
    "two": {
        "data": "data/input/spheres/two_spheres.npy",
        "operators": "results/trained_operators/two_spheres_N8.npz",
        "derived": "results/derived_quantities/two_spheres_N8.npz",
        "output": "figures/generated/figure_05_06_two_spheres",
        "figures": (5, 6), "centers": [(0,0,0), (0,0,3)], "radii": [1.5, 1.0],
        "monopoles": [[-.032258,-.077623,-.022179],[-.00011378,.042705,-.038174],
                      [.0079224,-.029492,-.038174],[.015959,.018639,-.046171],
                      [-.00011378,.0025956,3.0648],[.015959,.010618,2.9288],
                      [.048103,-.061580,1.5773]],
        "charges": [1,1,1,1,-1,-1,1],
    },
    "nonuniform": {
        "data": "data/input/spheres/sphere_nonuniform.npy",
        "operators": "results/trained_operators/sphere_nonuniform_N8.npz",
        "derived": "results/derived_quantities/sphere_nonuniform_N8.npz",
        "output": "figures/generated/figure_07_08_nonuniform_sphere",
        "figures": (7, 8), "centers": [(0,0,0)], "radii": [1.0],
        "monopoles": [[-.2013,.0873,.0949],[-.0646,-.1101,.1025],[-.0114,.0570,.1329],
                      [.0038,-.1177,.1937],[.0266,.1557,.1329],[.1101,-.0418,.1025],
                      [.1937,-.0722,.0646]],
        "charges": [-1]*7,
    },
}


def sphere_surfaces(ax, centers, radii):
    u=np.linspace(0,2*np.pi,70); v=np.linspace(0,np.pi,70)
    for center,radius in zip(centers,radii):
        x=radius*np.outer(np.cos(u),np.sin(v))+center[0]
        y=radius*np.outer(np.sin(u),np.sin(v))+center[1]
        z=radius*np.outer(np.ones_like(u),np.cos(v))+center[2]
        ax.plot_surface(x,y,z,color="gray",alpha=.12,edgecolor="none")


def style_3d(ax, points):
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect(np.ptp(points,axis=0))


def monopoles(ax, cfg):
    points=np.asarray(cfg["monopoles"]); charges=np.asarray(cfg["charges"])
    colors=np.where(charges>0,"red","blue")
    ax.scatter(*points.T,c=colors,s=55,edgecolor="black",linewidth=.5,zorder=20)


def save(fig, path):
    fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig)


def run(kind: str):
    cfg=CONFIG[kind]; root=Path(__file__).resolve().parents[2]
    data=np.load(root/cfg["data"]); operators=np.load(root/cfg["operators"])["operators"]
    fuzzy=np.load(root/cfg["derived"])["fuzzy_manifold"]
    output=root/cfg["output"]; output.mkdir(parents=True,exist_ok=True)
    first,second=cfg["figures"]

    fig=plt.figure(figsize=(10,10)); ax=fig.add_subplot(111,projection="3d")
    ax.scatter(*data.T,color="black",s=2); sphere_surfaces(ax,cfg["centers"],cfg["radii"]); style_3d(ax,data)
    save(fig,output/f"figure_{first:02d}a_input_data.png")

    # The notebook evaluates curvature at input points, then colors the saved
    # row-aligned QCML expectation-value cloud. Its Hamiltonian omits our 1/2.
    curvature=berry_curvature(data,operators)/4.0
    curvature_mag=np.linalg.norm(np.column_stack((curvature[:,1,2],curvature[:,2,0],curvature[:,0,1])),axis=1)
    fig=plt.figure(figsize=(12,12)); ax=fig.add_subplot(111,projection="3d")
    sc=ax.scatter(*fuzzy.T,c=np.log1p(curvature_mag),cmap="plasma",s=2)
    sphere_surfaces(ax,cfg["centers"],cfg["radii"]); monopoles(ax,cfg); style_3d(ax,fuzzy)
    cb=fig.colorbar(sc,ax=ax,shrink=.6,pad=.1); cb.set_label(r"$\log(1+|\omega|)$")
    save(fig,output/f"figure_{first:02d}b_qcml_cloud.png")

    # Author notebook: uniform points in a 1.5x-expanded data bounding box.
    rng=np.random.RandomState(42); ranges=[]
    for low,high in zip(data.min(0),data.max(0)):
        center=.5*(low+high); half=.75*(high-low); ranges.append((center-half,center+half))
    probes=np.column_stack([rng.uniform(low,high,50_000) for low,high in ranges])
    _,states=coherent_states(probes,operators); geometry=expectation_values(states,operators)
    sigma=np.sqrt(np.maximum(uncertainty(states,operators),0))
    fig=plt.figure(figsize=(15,12)); ax=fig.add_subplot(111,projection="3d")
    sc=ax.scatter(*geometry.T,c=sigma,cmap="viridis",s=1,alpha=.65)
    sphere_surfaces(ax,cfg["centers"],cfg["radii"]); monopoles(ax,cfg); style_3d(ax,geometry)
    cb=fig.colorbar(sc,ax=ax,shrink=.6,pad=.1); cb.set_label(r"$\sigma$")
    save(fig,output/f"figure_{second:02d}a_quantum_geometry.png")

    # The author HSM helper divides the double-commutator sum by D.
    spectrum=np.linalg.eigvalsh(matrix_laplacian(operators)/operators.shape[0]).real
    fig,ax=plt.subplots(figsize=(10,8)); ax.plot(np.arange(12),np.sort(spectrum)[:12],"o-")
    ax.set_xlabel("Eigenvalue index",fontsize=16); ax.set_ylabel(r"$\lambda$",fontsize=16)
    save(fig,output/f"figure_{second:02d}b_laplacian_spectrum.png")
    print(f"Wrote paper Figures {first} and {second} to {output}")
