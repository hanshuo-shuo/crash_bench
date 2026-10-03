"""Render every report page for human visual review, using the available MuPDF runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import fitz


def render(source, output):
    output.mkdir(parents=True, exist_ok=True)
    pages = []
    with fitz.open(source) as document:
        for index, page in enumerate(document):
            image_path = output / f'page_{index + 1:02d}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(image_path)
            text = page.get_text()
            (output / f'page_{index + 1:02d}.txt').write_text(text)
            outside = []
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        if not page.rect.contains(fitz.Rect(span['bbox'])):
                            outside.append(span['text'])
            pages.append(dict(page=index + 1, characters=len(text), outside_page=outside,
                              render=image_path.name))
    # Four pages per sheet keep page balance and breaks visible at once.
    with fitz.open(source) as document:
        for offset in range(0, len(pages), 4):
            with fitz.open() as contact:
                sheet = contact.new_page(width=1240, height=1640)
                for slot, record in enumerate(pages[offset:offset + 4]):
                    x = 16 + (slot % 2) * 620
                    y = 26 + (slot // 2) * 820
                    sheet.show_pdf_page(fitz.Rect(x, y, x + 588, y + 760), document, record['page'] - 1)
                    sheet.insert_text((x, y - 8), f"Page {record['page']}", fontsize=12)
                sheet.get_pixmap(alpha=False).save(output / f'contact_{offset // 4 + 1:02d}.png')
    receipt = dict(source=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   renderer='PyMuPDF ' + fitz.VersionBind, pages=pages,
                   visual_review_required=True,
                   scope='Bounds/text checks do not replace visual review of every rendered page.')
    (output / 'RENDER_READBACK.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
