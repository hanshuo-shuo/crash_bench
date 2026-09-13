"""Render the editable Markdown research manuscript as a readable review PDF."""
import html
from pathlib import Path
import re
import shutil
import subprocess

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'docs/iclr27/repeatable_value/manuscript.md'
OUTPUT = ROOT/'output/pdf/crashbench_repeatable_value_draft.pdf'
TEMP = ROOT/'tmp/pdfs'
FONT_ROOT = Path('/Users/hanshuo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')


def inline(text):
    text = html.escape(text.replace('—', '-').replace('–', '-').replace('−', '-'))
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" color="#206b75">\1</a>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<font name="Courier" size="8">\1</font>', text)
    return text


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#bac5ca'))
    canvas.line(48, 42, 564, 42)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#53636b'))
    canvas.drawString(48, 29, 'CrashBench | Research draft | 13 September 2026')
    canvas.drawRightString(564, 29, str(doc.page))
    canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    for face, filename in [('CBSerif', 'DejaVuSerif.ttf'), ('CBSerif-Bold', 'DejaVuSerif-Bold.ttf'),
                           ('CBSerif-Italic', 'DejaVuSerif-Italic.ttf'), ('CBSerif-BoldItalic', 'DejaVuSerif-BoldItalic.ttf')]:
        pdfmetrics.registerFont(TTFont(face, str(FONT_ROOT/filename)))
    pdfmetrics.registerFontFamily('CBSerif', normal='CBSerif', bold='CBSerif-Bold', italic='CBSerif-Italic', boldItalic='CBSerif-BoldItalic')
    styles = getSampleStyleSheet()
    body = ParagraphStyle('Body', fontName='CBSerif', fontSize=9.6, leading=13.9, spaceAfter=7,
                          textColor=colors.HexColor('#172932'), allowWidows=0, allowOrphans=0)
    title = ParagraphStyle('PaperTitle', parent=body, fontName='CBSerif-Bold', fontSize=19, leading=24,
                           spaceAfter=14, alignment=TA_CENTER)
    section = ParagraphStyle('Section', parent=body, fontName='CBSerif-Bold', fontSize=12.4, leading=17,
                             spaceBefore=12, spaceAfter=8, keepWithNext=True)
    sub = ParagraphStyle('Subsection', parent=section, fontSize=10.3, leading=14, spaceBefore=7)
    caption = ParagraphStyle('Caption', parent=body, fontSize=8, leading=11, textColor=colors.HexColor('#435962'))
    cell = ParagraphStyle('Cell', parent=body, fontName='Helvetica', fontSize=7.8, leading=10.2, spaceAfter=0)
    note = ParagraphStyle('Note', parent=caption, borderColor=colors.HexColor('#d3dde1'), borderWidth=.5,
                          borderPadding=8, backColor=colors.HexColor('#f0f5f6'), spaceAfter=12)
    lines = SOURCE.read_text().splitlines()
    flow = []
    i = 0
    equation = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('# '):
            flow.append(Paragraph(inline(line[2:]), title)); i += 1; continue
        if line.startswith('## '):
            flow.append(Paragraph(inline(line[3:]), section)); i += 1; continue
        if line.startswith('### '):
            flow.append(Paragraph(inline(line[4:]), sub)); i += 1; continue
        if line == r'\[':
            expr = []
            i += 1
            while lines[i].strip() != r'\]':
                expr.append(lines[i].strip()); i += 1
            equation += 1
            # Typeset math with the local TeX installation, then include it in the review PDF.
            target = TEMP/f'equation_{equation}.png'
            tex = TEMP/f'equation_{equation}.tex'
            tex.write_text('\\documentclass{article}\n\\usepackage{amsmath}\n\\pagestyle{empty}\n'
                           '\\newsavebox{\\eqbox}\n\\begin{document}\n'
                           '\\sbox{\\eqbox}{$\\displaystyle '+' '.join(expr)+'$}\n'
                           '\\pdfpagewidth=\\dimexpr\\wd\\eqbox+8pt\\relax\n'
                           '\\pdfpageheight=\\dimexpr\\ht\\eqbox+\\dp\\eqbox+8pt\\relax\n'
                           '\\pdfhorigin=4pt\\pdfvorigin=4pt\n'
                           '\\shipout\\vbox{\\offinterlineskip\\copy\\eqbox}\n\\end{document}\n')
            proc = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex.name],
                                  cwd=TEMP, text=True, capture_output=True)
            if proc.returncode:
                raise RuntimeError(proc.stdout[-3500:])
            converter = shutil.which('pdftoppm') or '/Users/hanshuo/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm'
            subprocess.run([converter, '-png', '-r', '220', '-singlefile', str(tex.with_suffix('.pdf')),
                            str(target.with_suffix(''))], check=True, capture_output=True)
            width, height = ImageReader(str(target)).getSize()
            display_width = min(490, width*72/220)
            flow.append(Image(str(target), width=display_width, height=display_width*height/width))
            flow.append(Spacer(1, 10)); i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row = [v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', v) for v in row):
                    rows.append(row)
                i += 1
            n = len(rows[0])
            widths = [78, 133, 52, 55, 53, 60, 85] if n == 7 else ([148, 80, 53, 53, 182] if n == 5 else [516/n]*n)
            rendered = [[Paragraph(('<b>'+inline(v)+'</b>') if r == 0 else inline(v), cell)
                         for v in row] for r, row in enumerate(rows)]
            table = Table(rendered, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f0f2')),
                ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#5a7f8a')),
                ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#9bb0b8')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8f9')]),
            ]))
            flow.extend([table, Spacer(1, 12)]); continue
        match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if match:
            path = (SOURCE.parent/match.group(2)).resolve()
            w, h = ImageReader(str(path)).getSize()
            image = Image(str(path), width=516, height=516*h/w)
            i += 1
            while i < len(lines) and not lines[i].strip(): i += 1
            cap = []
            if i < len(lines) and lines[i].startswith('Figure '):
                while i < len(lines) and lines[i].strip(): cap.append(lines[i].strip()); i += 1
            flow.append(KeepTogether([image, Spacer(1, 5), Paragraph(inline(' '.join(cap)), caption), Spacer(1, 10)]))
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '![')) and lines[i].strip() != r'\[':
            paragraph.append(lines[i].strip()); i += 1
        value = ' '.join(paragraph)
        style = note if value.startswith('Working research manuscript') else caption if value.startswith('Figure ') else body
        flow.append(Paragraph(inline(value), style))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(612, 792), leftMargin=48, rightMargin=48,
        topMargin=42, bottomMargin=55, title='Beyond Observed Rescue', author='CrashBench research draft')
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == '__main__':
    main()
