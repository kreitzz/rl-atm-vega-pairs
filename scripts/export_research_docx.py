"""Export the report's shared manuscript to editable Word without extra dependencies."""
from pathlib import Path
import json
import re
import struct
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
}
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
CONTENT = 'http://schemas.openxmlformats.org/package/2006/content-types'
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def node(parent, tag, attrs=None, text=None):
    prefix, local = tag.split(':')
    el = ET.SubElement(parent, f'{{{NS[prefix]}}}{local}', {
        (f'{{{NS[k.split(":")[0]]}}}{k.split(":")[1]}' if ':' in k else k): str(v)
        for k, v in (attrs or {}).items()
    })
    if text is not None:
        el.text = str(text)
    return el


def xml(root):
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def export(pages, charts, figure_numbers, out):
    """Keep every prose block, table and caption; embed the same audited PNGs."""
    document = ET.Element(f'{{{NS["w"]}}}document')
    body = node(document, 'w:body')
    relationships = ET.Element(f'{{{REL}}}Relationships')
    media = {}

    def relationship(kind, target, external=False):
        rid = f'rId{len(relationships)+1}'
        attrs = dict(Id=rid, Type=f'{NS["r"]}/{kind}', Target=target)
        if external:
            attrs['TargetMode'] = 'External'
        ET.SubElement(relationships, f'{{{REL}}}Relationship', attrs)
        return rid

    relationship('styles', 'styles.xml')
    header_id = relationship('header', 'header1.xml')
    footer_id = relationship('footer', 'footer1.xml')

    def run(parent, text, linked=False):
        r = node(parent, 'w:r')
        if linked:
            props = node(r, 'w:rPr')
            node(props, 'w:color', {'w:val': '2463A6'})
            node(props, 'w:u', {'w:val': 'single'})
        t = node(r, 'w:t', text=text)
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

    def paragraph(parent, text='', style='Normal', page_break=False):
        p = node(parent, 'w:p')
        props = node(p, 'w:pPr')
        node(props, 'w:pStyle', {'w:val': style})
        if page_break:
            node(props, 'w:pageBreakBefore')
        start = 0
        for match in re.finditer(r'https://\S+', text):
            run(p, text[start:match.start()])
            link = node(p, 'w:hyperlink', {'r:id': relationship('hyperlink', match.group(), True)})
            run(link, match.group(), linked=True)
            start = match.end()
        run(p, text[start:])
        return p

    for index, page in enumerate(pages):
        paragraph(body, page['title'], 'Title', page_break=index > 0)
        if page.get('subtitle'):
            paragraph(body, page['subtitle'], 'Subtitle')
        for block in page['blocks']:
            kind = block['type']
            if kind in ('paragraph', 'heading', 'equation'):
                style = {'paragraph': 'Normal', 'heading': 'Heading2', 'equation': 'Equation'}[kind]
                if kind == 'paragraph' and re.match(r'^\[\d', block['text']):
                    style = 'Reference'
                paragraph(body, block['text'], style)
            elif kind == 'table':
                table = node(body, 'w:tbl')
                props = node(table, 'w:tblPr')
                node(props, 'w:tblW', {'w:w': 10240, 'w:type': 'dxa'})
                node(props, 'w:tblLayout', {'w:type': 'fixed'})
                widths = block.get('widths') or [512/len(block['headers'])]*len(block['headers'])
                grid = node(table, 'w:tblGrid')
                for width in widths:
                    node(grid, 'w:gridCol', {'w:w': round(width*20)})
                for row_index, values in enumerate([block['headers']]+block['rows']):
                    row = node(table, 'w:tr')
                    row_props = node(row, 'w:trPr')
                    node(row_props, 'w:cantSplit')
                    if row_index == 0:
                        node(row_props, 'w:tblHeader')
                    for width, value in zip(widths, values):
                        cell = node(row, 'w:tc')
                        cell_props = node(cell, 'w:tcPr')
                        node(cell_props, 'w:tcW', {'w:w': round(width*20), 'w:type': 'dxa'})
                        node(cell_props, 'w:shd', {'w:fill': 'EDF2F7' if row_index == 0 else 'F8FAFC'})
                        paragraph(cell, str(value), 'TableHeader' if row_index == 0 else 'TableText')
                paragraph(body, '', 'TableGap')
            elif kind == 'chart':
                name = block['name']
                data = (out.parent/'figures'/f'{name}.png').read_bytes()
                width_px, height_px = struct.unpack('>II', data[16:24])
                height = min(block['height'], 512*height_px/width_px)
                width = height*width_px/height_px
                cx, cy = round(width*12700), round(height*12700)
                media[f'{name}.png'] = data
                rid = relationship('image', f'media/{name}.png')
                p = paragraph(body, '', 'Figure')
                drawing = node(node(p, 'w:r'), 'w:drawing')
                inline = node(drawing, 'wp:inline', {'distT': 0, 'distB': 0, 'distL': 0, 'distR': 0})
                node(inline, 'wp:extent', {'cx': cx, 'cy': cy})
                number = figure_numbers[name]
                node(inline, 'wp:docPr', {'id': number, 'name': f'Figure {number}', 'descr': charts[name]['caption']})
                graphic = node(inline, 'a:graphic')
                graphic_data = node(graphic, 'a:graphicData', {'uri': NS['pic']})
                picture = node(graphic_data, 'pic:pic')
                nv = node(picture, 'pic:nvPicPr')
                node(nv, 'pic:cNvPr', {'id': number, 'name': f'{name}.png'})
                node(nv, 'pic:cNvPicPr')
                fill = node(picture, 'pic:blipFill')
                node(fill, 'a:blip', {'r:embed': rid})
                node(node(fill, 'a:stretch'), 'a:fillRect')
                shape = node(picture, 'pic:spPr')
                transform = node(shape, 'a:xfrm')
                node(transform, 'a:off', {'x': 0, 'y': 0})
                node(transform, 'a:ext', {'cx': cx, 'cy': cy})
                node(node(shape, 'a:prstGeom', {'prst': 'rect'}), 'a:avLst')
                paragraph(body, f'Figure {number}. '+charts[name]['caption'], 'Caption')

    section = node(body, 'w:sectPr')
    node(section, 'w:headerReference', {'w:type': 'default', 'r:id': header_id})
    node(section, 'w:footerReference', {'w:type': 'default', 'r:id': footer_id})
    node(section, 'w:pgSz', {'w:w': 12240, 'w:h': 15840})
    node(section, 'w:pgMar', {'w:top': 1180, 'w:right': 1000, 'w:bottom': 980,
                             'w:left': 1000, 'w:header': 400, 'w:footer': 400, 'w:gutter': 0})

    styles = ET.Element(f'{{{NS["w"]}}}styles')
    for name, font, size, after, color, bold in [
        ('Normal', 'Cambria', 21, 120, '202832', False),
        ('Title', 'Calibri', 34, 160, '2463A6', True),
        ('Subtitle', 'Calibri', 22, 140, '66717E', False),
        ('Heading2', 'Calibri', 24, 140, '2463A6', True),
        ('Equation', 'Cambria', 20, 120, '2463A6', False),
        ('Figure', 'Calibri', 2, 0, '202832', False),
        ('Caption', 'Calibri', 16, 120, '66717E', False),
        ('TableHeader', 'Calibri', 18, 60, '2463A6', True),
        ('TableText', 'Calibri', 18, 60, '202832', False),
        ('TableGap', 'Calibri', 2, 100, '202832', False),
        ('Reference', 'Cambria', 18, 100, '202832', False),
    ]:
        style = node(styles, 'w:style', {'w:type': 'paragraph', 'w:styleId': name})
        node(style, 'w:name', {'w:val': name})
        props = node(style, 'w:pPr')
        node(props, 'w:spacing', {'w:after': after, 'w:line': 265 if name == 'Normal' else 240,
                                'w:lineRule': 'exact' if name == 'Normal' else 'auto'})
        if name in ('Title', 'Heading2', 'Figure', 'TableHeader'):
            node(props, 'w:keepNext')
        node(props, 'w:widowControl')
        if name == 'Figure':
            node(props, 'w:jc', {'w:val': 'center'})
        run_props = node(style, 'w:rPr')
        node(run_props, 'w:rFonts', {'w:ascii': font, 'w:hAnsi': font, 'w:eastAsia': font})
        node(run_props, 'w:sz', {'w:val': size})
        node(run_props, 'w:color', {'w:val': color})
        if bold:
            node(run_props, 'w:b')

    header = ET.Element(f'{{{NS["w"]}}}hdr')
    paragraph(header, 'MFE230Z  |  OPTIONS RELATIVE VALUE · 6 OCTOBER 2026', 'Caption')
    footer = ET.Element(f'{{{NS["w"]}}}ftr')
    p = paragraph(footer, 'Kreitzman & Yang · ', 'Caption')
    node(p, 'w:fldSimple', {'w:instr': 'PAGE'})
    types = ET.Element(f'{{{CONTENT}}}Types')
    for extension, mime in [('rels', 'application/vnd.openxmlformats-package.relationships+xml'),
                            ('xml', 'application/xml'), ('png', 'image/png')]:
        ET.SubElement(types, f'{{{CONTENT}}}Default', dict(Extension=extension, ContentType=mime))
    for part, kind in [('document', 'document.main'), ('styles', 'styles'),
                       ('header1', 'header'), ('footer1', 'footer')]:
        ET.SubElement(types, f'{{{CONTENT}}}Override', {
            'PartName': f'/word/{part}.xml',
            'ContentType': f'application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml'})
    root_rels = ET.Element(f'{{{REL}}}Relationships')
    ET.SubElement(root_rels, f'{{{REL}}}Relationship', dict(
        Id='rId1', Type=f'{NS["r"]}/officeDocument', Target='word/document.xml'))
    partial = out.with_name('.'+out.name)
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, content in [('word/document.xml', document), ('word/styles.xml', styles),
                              ('word/header1.xml', header), ('word/footer1.xml', footer),
                              ('word/_rels/document.xml.rels', relationships),
                              ('_rels/.rels', root_rels), ('[Content_Types].xml', types)]:
            archive.writestr(path, xml(content))
        for name, data in media.items():
            archive.writestr('word/media/'+name, data)
    verify(partial, pages, charts, figure_numbers)
    partial.replace(out)


def verify(path, pages, charts, figure_numbers):
    """Check editable content and that all embedded figure relationships resolve."""
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        document = ET.fromstring(archive.read('word/document.xml'))
        paragraphs = [''.join(p.itertext()) for p in document.findall('.//w:p', NS)]
        for page in pages:
            assert page['title'] in paragraphs
            for block in page['blocks']:
                if block['type'] in ('paragraph', 'heading', 'equation'):
                    assert block['text'] in paragraphs, block['text']
                elif block['type'] == 'chart':
                    name = block['name']
                    assert f'Figure {figure_numbers[name]}. '+charts[name]['caption'] in paragraphs
                    assert archive.read(f'word/media/{name}.png') == (path.parent/'figures'/f'{name}.png').read_bytes()
                elif block['type'] == 'table':
                    for row in [block['headers']]+block['rows']:
                        assert all(str(value) in paragraphs for value in row)
        assert len(document.findall('.//w:tbl', NS)) == 2
        assert len(document.findall('.//w:drawing', NS)) == 10
        assert len(document.findall('.//w:pageBreakBefore', NS)) == 9


if __name__ == '__main__':
    paper = ROOT/'paper'
    manifest = json.loads((paper/'source_manifest.json').read_text())
    export(json.loads((paper/'manuscript.json').read_text()), manifest['charts'],
           manifest['figure_numbers'], paper/'options_relative_value.docx')
    print('Verified editable report text, two tables and ten embedded figures.')
