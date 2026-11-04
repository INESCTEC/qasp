"""Render QASP's conceptual architecture. Run with QASP's Matplotlib dependency.

Writes qasp-architecture.svg and qasp-architecture.png beside this script.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(16,11.9),dpi=100)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set_xlim(0,1600);ax.set_ylim(1190,0);ax.axis('off')
fig.patch.set_facecolor('#f8fafc')
fit_labels=[]
ink='#14243b';muted='#4e6179';line='#62768c';blue='#1573b8';purple='#6a46ba';teal='#087f8c'
def text(x,y,s,size=20,color=ink,weight='normal',ha='left'):
    return ax.text(x,y,s,fontsize=size*.72,color=color,weight=weight,ha=ha,va='center')
def box(x,y,w,h,title,lines,color=blue,fill='white',title_size=24,body_size=19):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=14',linewidth=1.3,edgecolor='#ccd8e5',facecolor=fill,zorder=2))
    ax.plot([x+20,x+20],[y+23,y+h-23],color=color,lw=3,solid_capstyle='round',zorder=3)
    fit_labels.append((text(x+38,y+35,title,title_size,color,'bold'),w-58))
    for i,s in enumerate(lines): fit_labels.append((text(x+38,y+70+i*26,s,body_size,muted),w-58))
def arrow(points,color=line,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.8,linestyle='--' if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=13,lw=1.8,color=color,linestyle='--' if dashed else '-',zorder=1,shrinkA=0,shrinkB=2))
def section(y,num,title):
    text(48,y,num,16,blue,'bold');text(84,y,title,18,muted,'bold')
text(48,53,'QASP',40,ink,'bold')
text(202,54,'From reusable quantum building blocks to learning and optimization',26,muted)
text(48,92,'Conceptual architecture  ·  Arrows show composition and shared infrastructure, not a mandatory execution sequence.',17,muted)
section(137,'01','BUILDING BLOCKS')
box(48,164,338,154,'Feature maps',['AngleEmbedding','NeuralEmbedding','Input encoding + trainable scales'],teal,body_size=16.8)
box(428,164,360,154,'Ansatz templates',['AngleAnsatz  ·  IQPAnsatz','Feature map + gate structure','Layers · entanglers · reuploading'],blue,body_size=17)
box(830,164,722,154,'Parameter initialization',['Normal · Uniform · Beta · TruncatedNormal · LogNormal','Gamma · Exponential · Laplace · Rademacher','Constant · Zeros · Ones · Choice'],purple,body_size=18)
arrow([(386,240),(428,240)],teal)
arrow([(608,318),(608,392)])
text(622,350,'compose',16,muted)
arrow([(1005,318),(1005,350),(865,350),(865,392)],purple,True)
text(1040,350,'Ansatz weights + input scales / neural projections',16,purple)
section(370,'02','CONSTRUCTION AND EXECUTION')
box(48,392,290,144,'Readout',['Pauli expectations','Probabilities · variances','Observable / decoder choice'],teal,body_size=17)
box(378,392,650,144,'CircuitBuilder',['Ansatz + measurement → quantum function → QNode','Optional TorchLayer; configurable differentiation'],blue,fill='#edf5fc',body_size=20)
box(1068,392,484,144,'Device selection',['qasp.devices.create_device','device · device_kwargs · wires · shots'],purple,body_size=18)
arrow([(338,462),(378,462)],teal)
arrow([(1028,462),(1068,462)])
arrow([(1310,536),(1310,581)])
text(1292,560,'select backend',16,muted,ha='right')
# One execution band feeds application modules through their shared contracts.
ax.add_patch(FancyBboxPatch((48,584),1504,226,boxstyle='round,pad=0,rounding_size=16',facecolor='#edf1f8',edgecolor='#c8d4e3',linewidth=1.2,zorder=0))
text(74,613,'SIMULATORS AND EXECUTION BACKENDS',18,muted,'bold')
box(72,640,464,146,'IQPSimulator',['Commuting IQP circuits','Exact TN · sampled Van den Nest · auto','Torch gradients; VN multicore execution'],blue,body_size=18)
box(556,640,488,146,'PauliPropagationSimulator',['General supported Clifford / Pauli rotations','Exact by default; optional truncation','Differentiable Heisenberg propagation'],purple,title_size=23,body_size=18)
box(1064,640,464,146,'PennyLane devices',['default.qubit · Lightning · tensor networks','Other compatible installed plugins','Kokkos / MPI where supported'],teal,body_size=18)
text(800,842,'Shared circuit, readout, execution and gradient contracts',20,muted,ha='center')
section(880,'03','APPLICATION MODULES')
ax.plot([192,1408],[904,904],color=line,lw=1.6,zorder=1)
ax.plot([800,800],[856,904],color=line,lw=1.6,zorder=1)
apps=[('Reinforcement learning','qasp.rl',['Born / softmax policies','PQDQN Q-functions','REINFORCE · DQN · PPO'],blue),
      ('Classifiers','qasp.classifiers',['PQCClassifier','Expectation / Born readouts','Losses, training and metrics'],teal),
      ('Quantum kernels','qasp.qkernels',['Fidelity · projected · Fisher','Trainable kernels','Kernel-target alignment'],purple),
      ('Optimization','qasp.optimization',['QUBO → Ising','QAOA variants · PCE','Classical optimizer interfaces'],blue),
      ('Generative learning','qasp.genlearn',['PQCBM','MMD · NLL objectives','Training and sampling'],teal)]
for i,(title,module,lines,color) in enumerate(apps):
    x=48+i*304;arrow([(x+144,904),(x+144,928)])
    box(x,928,288,164,title,[module,*lines],color,title_size=20,body_size=16.5)
text(48,1129,'QAOA builds cost / mixer QNodes directly and shares device selection; PCE can use CircuitBuilder.',17,muted)
text(48,1158,'Backend eligibility depends on the circuit, readout and gradient method. Initializer support follows each model’s configuration.',17,muted)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for label,available in fit_labels:
    width=label.get_window_extent(renderer).width
    if width>available: label.set_fontsize(label.get_fontsize()*available/width*.98)
fig.canvas.draw()
assert all(label.get_window_extent(renderer).width<=available+.1 for label,available in fit_labels)
fig.savefig(OUT/'qasp-architecture.svg',facecolor=fig.get_facecolor(),metadata={'Title':'QASP architecture','Description':'Feature maps, ansatz templates, initialization, circuit construction, simulators and five application modules.'})
fig.savefig(OUT/'qasp-architecture.png',dpi=140,facecolor=fig.get_facecolor())
svg_path=OUT/'qasp-architecture.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
plt.close(fig)
print(OUT/'qasp-architecture.svg')
print(OUT/'qasp-architecture.png')
