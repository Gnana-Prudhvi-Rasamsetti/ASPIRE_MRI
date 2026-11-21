"""Generate scalable SVG figures from saved ASPIRE-MRI metrics (stdlib only).

Run from the project root: ``python -m evaluation.plot_results``.
No model loading, GPU use, or plotting package is required.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"dncnn": "#64748b", "unet": "#0ea5e9", "restormer": "#8b5cf6", "equal_weight": "#f59e0b", "adaptive": "#059669", "mse": "#0ea5e9", "mse_ssim": "#8b5cf6", "mse_ssim_gradient": "#f59e0b", "gaussian": "#0284c7", "rician": "#db2777", "mixed": "#d97706"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class Chart:
    def __init__(self, title: str, subtitle: str = "", width: int = 1200, height: int = 650):
        self.w, self.h = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<rect width="100%" height="100%" fill="#ffffff"/>',
                      '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1e293b}.muted{fill:#64748b}.grid{stroke:#dbe3ec;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}</style>',
                      f'<text x="{width/2}" y="42" text-anchor="middle" font-size="25" font-weight="700">{html.escape(title)}</text>']
        if subtitle:
            self.parts.append(f'<text x="{width/2}" y="70" text-anchor="middle" font-size="13" class="muted">{html.escape(subtitle)}</text>')

    def text(self, x, y, value, size=13, anchor="middle", color=None, weight=None, rotate=None):
        extra = (f' fill="{color}"' if color else "") + (f' font-weight="{weight}"' if weight else "")
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"{extra}{transform}>{html.escape(str(value))}</text>')

    def rect(self, x, y, w, h, fill, rx=2, stroke=None):
        edge = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0,w):.1f}" height="{max(0,h):.1f}" rx="{rx}" fill="{fill}"{edge}/>')

    def line(self, x1, y1, x2, y2, color="#64748b", width=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')

    def circle(self, x, y, r, fill, stroke=None):
        edge = f' stroke="{stroke}" stroke-width="2.5"' if stroke else ""
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{edge}/>')

    def axes(self, x, y, w, h, ticks, lo, hi, formatter):
        for t in ticks:
            yy = y + h - (t-lo)/(hi-lo)*h
            self.line(x, yy, x+w, yy, "#dbe3ec", 1)
            self.text(x-12, yy+4, formatter(t), 11, "end", "#64748b")
        self.line(x, y, x, y+h, "#64748b")
        self.line(x, y+h, x+w, y+h, "#64748b")
        return lambda v: y+h-(v-lo)/(hi-lo)*h

    def write(self, path: Path, footnote: str = ""):
        if footnote:
            self.text(self.w/2, self.h-18, footnote, 12, color="#64748b")
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def model_comparison(data, out):
    restored = data["checkpoints"]["frozen_adaptive"]["restoration"]
    models = list(restored)
    labels = ["DnCNN", "U-Net", "Restormer", "Equal-weight fusion", "Adaptive fusion"]
    metrics = [("rmse", "RMSE (×10⁻³), lower is better", 1000), ("psnr", "PSNR (dB), higher is better", 1), ("ssim", "SSIM, higher is better", 1)]
    c = Chart("Held-out test performance by restoration model", f"{data['subject_count']} test subjects · {data['checkpoints']['frozen_adaptive']['sample_count']:,} slices · frozen adaptive checkpoint")
    left, top, pw, ph, gap = 90, 145, 310, 390, 70
    for j, (metric, ylabel, scale) in enumerate(metrics):
        vals = [restored[m][metric]*scale for m in models]
        lo, hi = min(vals), max(vals)
        pad = (hi-lo)*.45 if hi != lo else abs(hi)*.08
        lo, hi = lo-pad, hi+pad
        ymap = c.axes(left+j*(pw+gap), top, pw, ph, np_ticks(lo,hi), lo,hi, lambda v: f"{v:.3f}" if metric=="ssim" else f"{v:.2f}")
        c.text(left+j*(pw+gap)+pw/2, 118, ylabel, 15, weight="700")
        slot = pw/len(models)
        for i,(m,label,val) in enumerate(zip(models,labels,vals)):
            x=left+j*(pw+gap)+i*slot+slot*.18; y=ymap(val)
            c.rect(x,y,slot*.64,top+ph-y,COLORS[m])
            c.text(x+slot*.32,y-9,f"{val:.3f}" if metric=="ssim" else f"{val:.2f}",10)
            c.text(x+slot*.32,top+ph+28,label,10,rotate=-22)
    c.write(out/"test_model_comparison.svg", "Metrics are aggregated as subject means; each metric uses its own vertical scale.")


def np_ticks(lo, hi, n=5):
    step=(hi-lo)/(n-1)
    return [lo+i*step for i in range(n)]


def loss_ablation(data, out):
    variants=list(data["variants"])
    labels=["MSE", "MSE + SSIM", "MSE + SSIM + gradient"]
    metrics=[("rmse","RMSE (×10⁻³)",1000), ("psnr","PSNR (dB)",1), ("ssim","SSIM",1)]
    c=Chart("U-Net loss ablation", "Validation drives variant selection; test split is shown for final reporting")
    left, top, pw, ph, gap=105,145,300,380,75
    for j,(metric,title,scale) in enumerate(metrics):
        vals=[data["variants"][v][split]["subject_mean"][metric]*scale for v in variants for split in ("validation","test")]
        lo,hi=min(vals),max(vals); pad=(hi-lo)*.35 or max(abs(hi)*.06,.01); lo-=pad;hi+=pad
        x0=left+j*(pw+gap); ymap=c.axes(x0,top,pw,ph,np_ticks(lo,hi),lo,hi,lambda v:f"{v:.3f}" if metric=="ssim" else f"{v:.2f}")
        c.text(x0+pw/2,118,title,15,weight="700")
        slot=pw/3; bw=slot*.27
        for i,(v,label) in enumerate(zip(variants,labels)):
            for k,split in enumerate(("validation","test")):
                value=data["variants"][v][split]["subject_mean"][metric]*scale
                x=x0+i*slot+slot*.2+k*(bw+5); y=ymap(value)
                c.rect(x,y,bw,top+ph-y,"#93c5fd" if split=="validation" else "#2563eb")
                c.text(x+bw/2,y-7,f"{value:.3f}" if metric=="ssim" else f"{value:.2f}",9)
            c.text(x0+i*slot+slot/2,top+ph+35,label,10,rotate=-15)
    c.rect(450,590,15,15,"#93c5fd"); c.text(474,603,"Validation",12,"start")
    c.rect(570,590,15,15,"#2563eb"); c.text(594,603,"Test",12,"start")
    c.write(out/"unet_loss_ablation.svg", "Subject-mean results. The held-out test set is reported, not used to choose the loss.")


def robustness(data,out):
    levels=[float(v) for v in data["severity_levels"]]
    c=Chart("Adaptive restoration robustness across noise severity", "Test subject-mean SSIM; filled circles = training severity, hollow circles = unseen/intermediate")
    left,top,pw,ph,gap=100,145,300,390,70
    seen={round(float(v),3) for v in data["seen_training_levels"]}
    for j,noise in enumerate(data["noise_types"]):
        x0=left+j*(pw+gap); lo,hi=.45,1.0
        ymap=c.axes(x0,top,pw,ph,[.5,.6,.7,.8,.9,1.0],lo,hi,lambda v:f"{v:.1f}")
        c.text(x0+pw/2,118,noise.capitalize(),16,color=COLORS[noise],weight="700")
        pts=[]
        for severity in levels:
            val=data["conditions"][f"{noise}_{severity:.3f}"]["subject_mean"]["ssim"]
            xx=x0+(severity-min(levels))/(max(levels)-min(levels))*pw
            yy=ymap(val); pts.append((xx,yy,severity,val))
        for a,b in zip(pts,pts[1:]): c.line(a[0],a[1],b[0],b[1],COLORS[noise],3)
        for xx,yy,severity,val in pts:
            c.circle(xx,yy,6,COLORS[noise] if round(severity,3) in seen else "white",None if round(severity,3) in seen else COLORS[noise])
            c.text(xx,yy-12,f"{val:.3f}",10,color=COLORS[noise])
            c.text(xx,top+ph+22,f"{severity:.2f}",10,color="#64748b")
        c.text(x0+pw/2,top+ph+50,"Noise severity (σ)",11)
    c.circle(405,600,6,"#059669"); c.text(421,605,"Seen during training",11,"start")
    c.circle(585,600,6,"white","#059669"); c.text(601,605,"Unseen/intermediate",11,"start")
    c.write(out/"robustness_ssim_by_severity.svg", "Results are descriptive test-set measurements and were not used for tuning.")


def efficiency(data,out):
    models=list(data["results"]); labels=["DnCNN","U-Net","Restormer","Adaptive fusion"]
    batches=[str(x) for x in data["batch_sizes"]]
    vals=[data["results"][m][b]["images_per_second"] for m in models for b in batches]
    hi=max(vals)*1.2
    c=Chart("GPU inference throughput",f"{data.get('device','GPU')} · images per second (higher is faster)",width=1000,height=650)
    left,top,pw,ph=100,140,800,400
    ymap=c.axes(left,top,pw,ph,[0,hi*.2,hi*.4,hi*.6,hi*.8,hi],0,hi,lambda v:f"{v:.0f}")
    slot=pw/len(models); bw=slot*.28
    fills=["#38bdf8","#2563eb"]
    for i,(m,label) in enumerate(zip(models,labels)):
        for k,b in enumerate(batches):
            val=data["results"][m][b]["images_per_second"]
            x=left+i*slot+slot*.2+k*(bw+8); y=ymap(val)
            c.rect(x,y,bw,top+ph-y,fills[k]); c.text(x+bw/2,y-8,f"{val:.0f}",11)
        c.text(left+i*slot+slot/2,top+ph+30,label,12)
    for i,b in enumerate(batches):
        c.rect(375+i*160,600,16,16,fills[i]); c.text(400+i*160,613,f"Batch size {b}",12,"start")
    c.write(out/"inference_throughput.svg", "Throughput is calculated as batch size divided by measured batch latency.")


def detector_confusion(data, out):
    cp=data["checkpoints"]["frozen_adaptive"]
    mat=cp["noise_characterization"]["confusion_matrix"]
    names=["Gaussian","Rician","Mixed"]
    vmax=max(max(row) for row in mat)
    c=Chart("Noise detector confusion matrix",f"Held-out test slices · accuracy {cp['noise_characterization']['accuracy']*100:.3f}% · rows = true class, columns = predicted class",width=900,height=700)
    x0,y0,cell=270,155,125
    for i,name in enumerate(names):
        c.text(x0-cell/2,y0+i*cell+cell/2+5,name,15,"end")
        c.text(x0+i*cell+cell/2,y0-20,name,15)
    for i,row in enumerate(mat):
        for j,count in enumerate(row):
            intensity=count/vmax if vmax else 0
            fill="#0f766e" if i==j else "#e2e8f0"
            if i==j:
                # Tint diagonal by count, retaining readable white labels.
                fill=f"rgb({int(15+30*(1-intensity))},{int(118+35*(1-intensity))},{int(110+35*(1-intensity))})"
            c.rect(x0+j*cell,y0+i*cell,cell-4,cell-4,fill,rx=4)
            c.text(x0+j*cell+(cell-4)/2,y0+i*cell+(cell-4)/2+6,f"{count:,}",20,color="white" if i==j else "#334155",weight="700")
    c.text(x0+cell*1.5,y0+cell*3+45,"Predicted class",15,weight="700")
    c.text(100,y0+cell*1.5,"True class",15,rotate=-90,weight="700")
    c.write(out/"noise_detector_confusion_matrix.svg", "Only six test-slice errors: one Rician → mixed and five mixed → Rician.")


def gate_weights(data, out):
    cp=data["checkpoints"]["frozen_adaptive"]
    weights=cp["mean_gate_weights_by_noise_class"]
    classes=["Gaussian","Rician","Mixed"]
    experts=[("DnCNN","#64748b"),("U-Net","#0ea5e9"),("Restormer","#8b5cf6")]
    c=Chart("Adaptive fusion weights by true noise class","Mean gate weights on held-out test slices · each row sums to one",width=1000,height=650)
    x0,y0,w,h=290,160,600,105
    for i,cls in enumerate(classes):
        c.text(x0-20,y0+i*h+h/2+5,cls,17,"end",weight="700")
        row=weights[str(i)]; x=x0
        for j,(expert,color) in enumerate(experts):
            sw=w*row[j]
            c.rect(x,y0+i*h,sw,h-14,color,rx=4)
            if sw>52: c.text(x+sw/2,y0+i*h+(h-14)/2+5,f"{row[j]:.1%}",14,color="white",weight="700")
            x+=sw
    for j,(expert,color) in enumerate(experts):
        c.rect(290+j*190,535,18,18,color);c.text(318+j*190,550,expert,13,"start")
    c.write(out/"adaptive_gate_weights_by_noise.svg","Weights describe routing behavior; they do not by themselves establish which expert caused the final quality.")


def robustness_heatmap(data,out):
    noises=data["noise_types"]
    levels=[float(x) for x in data["severity_levels"]]
    vals=[[data["conditions"][f"{noise}_{severity:.3f}"]["subject_mean"]["ssim"] for severity in levels] for noise in noises]
    low=min(map(min,vals)); high=max(map(max,vals))
    c=Chart("Robustness SSIM overview","Adaptive model · test-set subject means across all corruption types and severity levels",width=1100,height=600)
    x0,y0,cw,ch=270,155,115,90
    for j,sev in enumerate(levels): c.text(x0+j*cw+cw/2,y0-20,f"σ = {sev:.3f}",13)
    for i,noise in enumerate(noises):
        c.text(x0-20,y0+i*ch+ch/2+5,noise.capitalize(),15,"end",color=COLORS[noise],weight="700")
        for j,val in enumerate(vals[i]):
            t=(val-low)/(high-low) if high>low else 0
            # blue-to-teal scale; strong SSIM is darker
            red=int(219-180*t); green=int(234-80*t); blue=int(254-55*t)
            fill=f"rgb({red},{green},{blue})"
            c.rect(x0+j*cw,y0+i*ch,cw-3,ch-3,fill,rx=4)
            c.text(x0+j*cw+cw/2,y0+i*ch+ch/2+5,f"{val:.3f}",15,color="#ffffff" if t>.60 else "#1e293b",weight="700")
    c.write(out/"robustness_ssim_heatmap.svg",f"Color scale spans observed SSIM values ({low:.3f}–{high:.3f}); higher is better.")


def audit_input_restored(data,out):
    conditions=data["conditions"]
    severities=[float(v) for v in data["severity_levels"]]
    c=Chart("SSIM before and after adaptive restoration","Independent-seed audit · subject-mean SSIM on the held-out test subjects",width=1200,height=680)
    left,top,pw,ph,gap=105,150,300,390,70
    colors={"noisy":"#94a3b8","restored":"#059669"}
    for j,noise in enumerate(data["noise_types"]):
        x0=left+j*(pw+gap); lo,hi=.35,1.0
        ymap=c.axes(x0,top,pw,ph,[.4,.5,.6,.7,.8,.9,1.0],lo,hi,lambda v:f"{v:.1f}")
        c.text(x0+pw/2,122,noise.capitalize(),16,color=COLORS[noise],weight="700")
        for metric,color in colors.items():
            pts=[]
            for sev in severities:
                item=conditions[f"independent_{noise}_{sev:.3f}"]["subject_mean"]
                ssim=item[metric]["ssim"]
                xx=x0+(sev-min(severities))/(max(severities)-min(severities))*pw
                yy=ymap(ssim);pts.append((xx,yy,sev,ssim))
            for a,b in zip(pts,pts[1:]):c.line(a[0],a[1],b[0],b[1],color,3)
            for xx,yy,sev,val in pts:
                c.circle(xx,yy,5.5,color)
                c.text(xx,yy-11,f"{val:.3f}",9,color=color)
        for sev in severities:
            xx=x0+(sev-min(severities))/(max(severities)-min(severities))*pw
            c.text(xx,top+ph+22,f"{sev:.2f}",10,color="#64748b")
        c.text(x0+pw/2,top+ph+48,"Noise severity (σ)",11)
    c.line(380,610,415,610,colors["noisy"],3);c.text(424,615,"Noisy input",12,"start")
    c.line(560,610,595,610,colors["restored"],3);c.text(604,615,"Restored",12,"start")
    c.write(out/"ssim_noisy_vs_restored_audit.svg","Restoration improves SSIM in every audited condition; values are descriptive test results, not tuning data.")


def line_panels(title, subtitle, y_label, series_for_panel, x_levels, output, footnote, y_range=None, series_colors=None):
    c=Chart(title,subtitle,width=1200,height=680)
    left,top,pw,ph,gap=105,150,300,390,70
    palette=series_colors or COLORS
    for j,(panel_title,series) in enumerate(series_for_panel):
        x0=left+j*(pw+gap)
        all_y=[float(v) for _,ys,_ in series for v in ys]
        lo,hi=y_range if y_range else (min(all_y),max(all_y))
        pad=(hi-lo)*.08 if hi>lo else .02
        lo=max(0,lo-pad) if y_range is None else lo
        hi=hi+pad if y_range is None else hi
        ymap=c.axes(x0,top,pw,ph,np_ticks(lo,hi,5),lo,hi,lambda v:f"{v:.2f}")
        c.text(x0+pw/2,122,panel_title,16,weight="700")
        for series_name,ys,color_key in series:
            color=palette.get(color_key,color_key)
            pts=[]
            for xv,yv in zip(x_levels,ys):
                xx=x0+(float(xv)-min(x_levels))/(max(x_levels)-min(x_levels))*pw
                yy=ymap(float(yv));pts.append((xx,yy,float(xv),float(yv)))
            for a,b in zip(pts,pts[1:]):c.line(a[0],a[1],b[0],b[1],color,2.8)
            for xx,yy,xv,yv in pts:
                c.circle(xx,yy,4.5,color)
                c.text(xx,yy-10,f"{yv:.3f}" if abs(yv)<1 else f"{yv:.2f}",9,color=color)
        for xv in x_levels:
            xx=x0+(float(xv)-min(x_levels))/(max(x_levels)-min(x_levels))*pw
            c.text(xx,top+ph+21,f"{float(xv):.2f}",10,color="#64748b")
        c.text(x0+pw/2,top+ph+47,"Noise severity (σ)",11)
    c.text(600,top+ph+79,y_label,13,weight="700")
    # Shared legend; lay items out in one row.
    legend_items=series_for_panel[0][1]
    total=len(legend_items); spacing=min(230,880/max(total,1)); start=(1200-(spacing*total))/2
    for i,(name,_,color_key) in enumerate(legend_items):
        x=start+i*spacing
        color=palette.get(color_key,color_key)
        c.line(x,605,x+25,605,color,3);c.circle(x+12,605,4,color)
        c.text(x+32,610,name,11,"start")
    c.write(output,footnote)


def robustness_metric_lines(data,out,metric,filename,title,y_label,scale=1.0):
    levels=[float(v) for v in data["severity_levels"]]
    panels=[]
    for noise in data["noise_types"]:
        ys=[data["conditions"][f"{noise}_{sev:.3f}"]["subject_mean"][metric]*scale for sev in levels]
        panels.append((noise.capitalize(),[(noise.capitalize(),ys,noise)]))
    line_panels(title,"Frozen adaptive model · held-out test subject means",y_label,panels,levels,out/filename,"Test-set trends are descriptive; the checkpoint was selected on validation.")


def gate_weight_lines(data,out):
    levels=[float(v) for v in data["severity_levels"]]
    panels=[]
    expert_names=["DnCNN","U-Net","Restormer"]
    expert_colors={"DnCNN":"#64748b","U-Net":"#0ea5e9","Restormer":"#8b5cf6"}
    for noise in data["noise_types"]:
        rows=[data["conditions"][f"{noise}_{sev:.3f}"]["mean_gate_weights"] for sev in levels]
        series=[(name,[row[i] for row in rows],name) for i,name in enumerate(expert_names)]
        panels.append((noise.capitalize(),series))
    line_panels("Adaptive gate weights vs. noise severity","Mean routing weight over test slices · one panel per corruption type","Mean expert weight",panels,levels,out/"adaptive_gate_weights_vs_severity.svg","Routing weights sum to one at each condition; they indicate gate behavior, not causal expert importance.",y_range=(0,1),series_colors=expert_colors)


def audit_ssim_lines(data,out):
    levels=[float(v) for v in data["severity_levels"]]
    panels=[]
    series_colors={"Noisy input":"#94a3b8","Restored":"#059669"}
    for noise in data["noise_types"]:
        series=[]
        for name,key in (("Noisy input","noisy"),("Restored","restored")):
            ys=[data["conditions"][f"independent_{noise}_{sev:.3f}"]["subject_mean"][key]["ssim"] for sev in levels]
            series.append((name,ys,name))
        panels.append((noise.capitalize(),series))
    line_panels("Noisy input vs. restored SSIM","Independent-seed audit · held-out test subject means","SSIM",panels,levels,out/"noisy_vs_restored_ssim_by_severity.svg","The audit includes σ = 0.02, 0.03, and 0.05; results were not used for tuning.",y_range=(0,1),series_colors=series_colors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results"/"final_figures")
    parser.add_argument("--all-figures",action="store_true",help="Also generate the four supplementary diagnostic plots.")
    parser.add_argument("--line-graphs-only",action="store_true",help="Generate four additional severity trend line graphs only.")
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    count=0
    if not args.line_graphs_only:
        model_comparison(read_json(args.results_dir/"test_comparison.json"),args.output_dir)
        loss_ablation(read_json(args.results_dir/"loss_ablation_unet.json"),args.output_dir)
        robustness(read_json(args.results_dir/"robustness_test.json"),args.output_dir)
        efficiency(read_json(args.results_dir/"efficiency.json"),args.output_dir)
        count=4
    if args.all_figures and not args.line_graphs_only:
        test=read_json(args.results_dir/"test_comparison.json")
        detector_confusion(test,args.output_dir)
        gate_weights(test,args.output_dir)
        robustness_heatmap(read_json(args.results_dir/"robustness_test.json"),args.output_dir)
        audit_input_restored(read_json(args.results_dir/"robustness_ssim_audit.json"),args.output_dir)
        count=8
    if args.line_graphs_only:
        robustness_data=read_json(args.results_dir/"robustness_test.json")
        robustness_metric_lines(robustness_data,args.output_dir,"rmse","rmse_vs_noise_severity.svg","Restoration RMSE vs. noise severity","RMSE (×10⁻³)",scale=1000)
        robustness_metric_lines(robustness_data,args.output_dir,"psnr","psnr_vs_noise_severity.svg","Restoration PSNR vs. noise severity","PSNR (dB)")
        audit_ssim_lines(read_json(args.results_dir/"robustness_ssim_audit.json"),args.output_dir)
        gate_weight_lines(robustness_data,args.output_dir)
        count=4
    print(f"Saved {count} scalable SVG figures to {args.output_dir}")


if __name__=="__main__": main()
