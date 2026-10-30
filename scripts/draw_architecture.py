"""Draw the structural learning pipeline and the two coupled denoising branches."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parents[1]/'results'/'figures'
P.mkdir(exist_ok=True,parents=True)


def node(ax,x,y,w,h,label,color='#e7eef7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=.035',
                                linewidth=1,edgecolor='#46647a',facecolor=color))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)


def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,
                                 linewidth=1.4,color='#506277'))


def save(fig,name):
    fig.savefig(P/(name+'.png'),dpi=240,bbox_inches='tight')
    fig.savefig(P/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(12,4));ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
node(ax,.2,1.8,2,.9,'DesignSafe archive\n621 frame designs')
node(ax,2.8,1.8,2,.9,'Geometry and\nW-section parsing')
node(ax,5.4,1.8,2,.9,'Hierarchical graph\n57 grouped tokens')
node(ax,8,1.8,1.7,.9,'Grouped train /\nvalidation / test')
node(ax,10.2,1.8,1.6,.9,'Model inputs\nand labels','#e2f0e6')
for x1,x2 in [(2.2,2.8),(4.8,5.4),(7.4,8.0),(9.7,10.2)]:arrow(ax,x1,2.25,x2,2.25)
ax.text(6,3.4,'Building-wise data separation; record-wise ground-motion split',ha='center',fontsize=13,weight='bold')
ax.text(6,.75,'Response matrices supply surrogate targets; only design rows supply generator targets',ha='center',fontsize=11)
save(fig,'data_processing_architecture')

fig,ax=plt.subplots(figsize=(11,6));ax.set(xlim=(0,11),ylim=(0,6));ax.axis('off')
node(ax,.4,4.2,2,.9,'Geometry + target\nseismic performance')
node(ax,3.1,4.2,2.1,.9,'Hierarchical graph\ncondition encoder')
node(ax,6.1,4.2,2.3,.9,'Shared diffusion\nTransformer blocks')
node(ax,5.7,2.25,1.9,.9,'Continuous Gaussian\ndenoising','#eaf0fb')
node(ax,8.3,2.25,1.9,.9,'Categorical W-section\ndenoising','#f0eaf9')
node(ax,6.8,.45,2.3,.9,'Section-property\nconsistency loss','#e2f0e6')
arrow(ax,2.4,4.65,3.1,4.65);arrow(ax,5.2,4.65,6.1,4.65)
arrow(ax,7.1,4.2,6.65,3.15);arrow(ax,7.7,4.2,9.25,3.15)
arrow(ax,6.65,2.25,7.45,1.35);arrow(ax,9.25,2.25,8.45,1.35)
ax.annotate('Cross-branch attention',xy=(8.6,3.2),xytext=(7.15,3.65),fontsize=10,
            arrowprops=dict(arrowstyle='->',color='#506277'))
ax.text(5.5,5.65,'Graph-conditioned discrete–continuous diffusion model',ha='center',fontsize=14,weight='bold')
save(fig,'diffusion_model_architecture')

fig,ax=plt.subplots(figsize=(11,5));ax.set(xlim=(0,11),ylim=(0,5));ax.axis('off')
node(ax,.35,2.2,2,.9,'Generated W-section\nframe candidates')
node(ax,2.95,2.2,2,.9,'Differentiable\nresponse surrogate')
node(ax,5.55,2.2,2.1,.9,'Empirical CVaR\nand design penalties')
node(ax,8.3,2.2,2.1,.9,'Risk-gradient\nguided denoising','#e2f0e6')
for a,b in [(2.35,2.95),(4.95,5.55),(7.65,8.3)]:arrow(ax,a,2.65,b,2.65)
node(ax,3.1,.45,2.5,.85,'OpenSees nonlinear analysis\n(external validation)','#f8efd8')
node(ax,6.25,.45,2.5,.85,'Uncertainty-driven\nactive-learning update','#f8efd8')
arrow(ax,9.35,2.2,7.5,1.3);arrow(ax,6.25,.87,5.6,.87)
arrow(ax,4.35,1.3,3.95,2.2)
ax.text(5.5,4.45,'Physical risk guidance and high-fidelity feedback architecture',ha='center',fontsize=14,weight='bold')
ax.text(5.5,3.8,'OpenSees feedback requires completed nonlinear simulations; it is not inferred from proxy predictions.',ha='center',fontsize=10)
save(fig,'seismic_risk_feedback_architecture')
