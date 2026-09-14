"""Render the research manuscript with the official ICLR 2027 style (no data analysis)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'docs/iclr27/repeatable_value/manuscript.md'
LATEX = SOURCE.parent/'latex'
BUILD = ROOT/'tmp/pdfs/iclr'
OUTPUT = ROOT/'output/pdf/crashbench_repeatable_value_draft.pdf'

CITES = {
    '2506.09937':'gu2025safe', '2609.05178':'liu2026recover',
    '2606.09258':'shin2026b2ff', '2608.14822':'zhang2026core',
    'sharma25a':'sharma2025decision', '2309.10874':'vincent2023guarantees',
    '2603.09292':'dai2026spr', '2406.09246':'kim2024openvla',
    '2410.24164':'black2024pi0', '2306.03310':'liu2023libero',
    '2606.07723':'chen2026volo',
}


def escape(value):
    value = value.replace('−','-').replace('–','-').replace('—','-')
    value = value.replace('→',r'$\rightarrow$').replace('≤',r'$\leq$')
    mapping = {'&':r'\&','%':r'\%','_':r'\_','#':r'\#','{':r'\{','}':r'\}'}
    return ''.join(mapping.get(c,c) for c in value)


def inline(value):
    tokens = []
    def token(s):
        tokens.append(s)
        return f'ZZTOKEN{len(tokens)-1}ZZ'
    def link(m):
        label,url = m.groups()
        key = next((v for k,v in CITES.items() if k in url),None)
        if key: return token(escape(label)+r'~\citep{'+key+'}')
        return token(r'\href{'+url.replace('%',r'\%')+'}{'+escape(label)+'}')
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,value)
    value = re.sub(r'`([^`]+)`',lambda m:token(r'\texttt{'+escape(m[1])+'}'),value)
    value = re.sub(r'\*\*(.+?)\*\*',lambda m:token(r'\textbf{'+escape(m[1])+'}'),value)
    value = re.sub(r'\bpi0\b',lambda m:token(r'$\pi_0$'),value)
    for pattern,replacement in [
        (r'\bOmega_X\b',r'$\Omega_X$'),(r'\bOmega\b',r'$\Omega$'),
        (r'\bdelta_A\b',r'$\delta_A$'),
    ]:
        value=re.sub(pattern,lambda m:token(replacement),value)
    value=re.sub(r'\bG_[ABC]\b',lambda m:token('$'+m[0]+'$'),value)
    value=re.sub(r'\b[qY]_[aBR]\(s,h\)',lambda m:token('$'+m[0]+'$'),value)
    value = escape(value)
    for i,t in enumerate(tokens): value = value.replace(f'ZZTOKEN{i}ZZ',t)
    return value


def convert(markdown):
    lines=markdown.splitlines(); result=[]; i=0; abstract=False; appendix=False
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line.startswith('# '): i+=1;continue
        if line.startswith('Working research manuscript'):
            while i<len(lines) and not lines[i].startswith('## '): i+=1
            continue
        if line=='## Abstract':
            result.append(r'\begin{abstract}');abstract=True;i+=1;continue
        if line.startswith('## '):
            if abstract: result.append(r'\end{abstract}');abstract=False
            title=re.sub(r'^\d+\.\s*','',line[3:])
            if title.startswith('Appendix') and not appendix:
                result.append(r'\label{endofmain}')
                result.append(r'\input{statements.tex}')
                result.append(r'\bibliography{references}\bibliographystyle{iclr2027_conference}')
                result.append(r'\clearpage\appendix');appendix=True
            title=re.sub(r'^Appendix\s+[A-Z]\.?\s*','',title)
            result.append(r'\section{'+inline(title)+'}');i+=1;continue
        if line.startswith('### '):
            title=re.sub(r'^\d+\.\d+\s*','',line[4:])
            result.append(r'\subsection{'+inline(title)+'}');i+=1;continue
        if line==r'\[':
            math=[];i+=1
            while lines[i].strip()!=r'\]':math.append(lines[i].strip());i+=1
            eq=' '.join(math)
            if r'\qquad' in eq and len(eq)>100:
                eq=r'\begin{gathered}'+eq.replace(r',\qquad',r',\\')+r'\end{gathered}'
            result.append(r'\begin{equation}'+eq+r'\end{equation}');i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-:]+',s) for s in row):table.append(row)
                i+=1
            n=len(table[0]);spec=' '.join([r'>{\raggedright\arraybackslash}X']*n)
            result += [r'\begin{center}\small\setlength{\tabcolsep}{3pt}',
                r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule']
            for j,row in enumerate(table):
                result.append(' & '.join((r'\textbf{'+inline(s)+'}') if j==0 else inline(s) for s in row)+r' \\')
                if j==0: result.append(r'\midrule')
            result += [r'\bottomrule\end{tabularx}\end{center}'];continue
        if line.startswith('!['):
            match=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
            path=(SOURCE.parent/match[2]).resolve();i+=1
            while i<len(lines) and not lines[i].strip():i+=1
            caption=[]
            if i<len(lines) and lines[i].startswith('Figure '):
                while i<len(lines) and lines[i].strip():caption.append(lines[i].strip());i+=1
            text=re.sub(r'^Figure\s+\d+\.\s*','',' '.join(caption)) or match[1]
            # Prefer the version rendered at conference-column size when available.
            replacement=SOURCE.parent/'figures'/path.name
            if replacement.exists():path=replacement
            rel=os.path.relpath(path,LATEX)
            result += [r'\begin{figure}[tb]\centering',
                r'\includegraphics[width=\linewidth]{'+rel+'}',
                r'\caption{'+inline(text)+'}',r'\end{figure}'];continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','\\[')):
            para.append(lines[i].strip());i+=1
        result.append(inline(' '.join(para))+'\n')
    if abstract:result.append(r'\end{abstract}')
    if not appendix:
        result += [r'\label{endofmain}',r'\input{statements.tex}',
            r'\bibliography{references}\bibliographystyle{iclr2027_conference}']
    return '\n\n'.join(result).rstrip()+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-copy',action='store_true');args=parser.parse_args()
    BUILD.mkdir(parents=True,exist_ok=True)
    source=SOURCE.read_text()
    title=source.splitlines()[0].removeprefix('# ')
    display_title=inline(title).replace('Evaluating Repeatable',r'Evaluating\\ Repeatable')
    (LATEX/'body.tex').write_text(convert(source))
    wrapper=r'''\documentclass{article}
\usepackage{iclr2027_conference,times}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,hyperref,url}
\urlstyle{same}
\renewcommand{\ttdefault}{cmtt}
\hypersetup{colorlinks=true,allcolors=blue,pdfauthor={},pdftitle={TITLE}}
\title{DISPLAYTITLE}
% The draft uses the official page dimensions and fonts; it is not a submission.
\author{Anonymous working draft}
\iclrfinalcopy
\begin{document}
\maketitle
\lhead{Working draft in ICLR 2027 format; not submitted}
\input{body.tex}
\end{document}
'''.replace('DISPLAYTITLE',display_title).replace('TITLE',inline(title))
    (LATEX/'main.tex').write_text(wrapper)
    env=dict(os.environ)
    env['TEXMFVAR']=str(ROOT/'tmp/pdfs/texmf-var')
    env['TEXINPUTS']=str(LATEX)+'//:'+str(ROOT/'tmp/pdfs/eso-pic')+'//:'+env.get('TEXINPUTS','')
    env['BIBINPUTS']=str(LATEX)+'//:'+env.get('BIBINPUTS','')
    env['BSTINPUTS']=str(LATEX)+'//:'+env.get('BSTINPUTS','')
    for index, command in enumerate([
        ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'main.tex'],
        ['bibtex','main'],
        ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'main.tex'],
        ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'main.tex'],
    ]):
        p=subprocess.run(command,cwd=BUILD if command[0]=='bibtex' else LATEX,
                         env=env,capture_output=True,text=True)
        (BUILD/f'build-{index}.log').write_text(p.stdout+'\n'+p.stderr)
        if p.returncode: raise RuntimeError((p.stdout+'\n'+p.stderr)[-4500:])
    if not args.no_copy:
        OUTPUT.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BUILD/'main.pdf',OUTPUT)
    print(json.dumps(dict(pdf=str(BUILD/'main.pdf' if args.no_copy else OUTPUT),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())))


if __name__=='__main__':main()
