"""Traditional academic typography for the integrated options manuscript.

Keeps the shared text and results; removes report branding and table shading.
Figure placements also support vector composition in the final PDF.
"""
import re

TITLE='Learning to Allocate Risk in Options Markets'
SUBTITLE='Contextual Bandits for Trade Sizing and Q-Learning for Capital Allocation'
AUTHORS='Adam Kreitzman and Soo Xin Yang'
COURSE=''
DATE='20 October 2026'
STYLE_VERSION='academic-serif-vector-v1'
INK='#111111'
BODY_SIZE=11
BODY_LEADING=13.6
FALLBACK_CHARS='₁₂'


def page_class(base):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from PIL import Image

    class AcademicPage(base):
        def __init__(self,number,title):
            self.fig=plt.figure(figsize=(8.5,11),facecolor='white')
            self.renderer=self.fig.canvas.get_renderer()
            self.x,self.width,self.y=54,504,738
            self.number=number
            self.in_abstract=False
            self.figure_placements=[]
            self.figure_axes=[]
            if number==1:
                font=FontProperties(family='STIXGeneral',size=21,weight='bold')
                width=self.renderer.get_text_width_height_descent(TITLE,font,ismath=False)[0]*72/self.fig.dpi
                assert width<=self.width,('Title must stay on one line',width)
                self.center(726,TITLE,21,'bold')
                self.center(699,SUBTITLE,12)
                self.center(672,AUTHORS,11.5)
                self.center(654,COURSE,10)
                self.center(636,DATE,10)
                self.y=608
            else:
                self.heading(title,size=15)
            self.center(31,str(number),9)

        def center(self,y,text,size=11,weight='normal'):
            return self.fig.text(.5,y/792,text,fontsize=size,color=INK,
                    family='STIXGeneral',weight=weight,ha='center',va='top',parse_math=False)

        def text(self,x,y,text,size=BODY_SIZE,color=INK,family='STIXGeneral',weight='normal'):
            family=['STIXGeneral','DejaVu Serif'] if any(c in text for c in FALLBACK_CHARS) else 'STIXGeneral'
            return self.fig.text(x/612,y/792,text,fontsize=size,color=INK,
                    family=family,weight=weight,va='top',parse_math=False)

        def lines(self,text,size,family='STIXGeneral',weight='normal',width=None):
            family=['STIXGeneral','DejaVu Serif'] if any(c in text for c in FALLBACK_CHARS) else 'STIXGeneral'
            return super().lines(text,size,family,weight,width)

        def paragraph(self,text,size=BODY_SIZE,color=INK,leading=BODY_LEADING):
            if self.in_abstract:
                old_x,old_width=self.x,self.width
                self.x+=24;self.width-=48
                super().paragraph(text,size=10.2,color=INK,leading=12.7)
                self.x,self.width=old_x,old_width
                self.y-=5;self.in_abstract=False
            else:
                super().paragraph(text,size=size,color=INK,leading=leading)

        def heading(self,text,size=12):
            if text=='Abstract' and self.number==1:
                self.center(self.y,text,11,'bold');self.y-=22;self.in_abstract=True
                return
            self.y-=3
            for line in self.lines(text,size,weight='bold'):
                self.text(self.x,self.y,line,size,weight='bold');self.y-=size*1.25
            self.y-=9

        def rule(self,y,width=.55):
            self.fig.add_artist(plt.Line2D([self.x/612,(self.x+self.width)/612],
                [y/792,y/792],transform=self.fig.transFigure,color=INK,linewidth=width))

        def table(self,block):
            raw=block.get('widths') or [1]*len(block['headers'])
            widths=[w/sum(raw)*self.width for w in raw]
            numeric=[False]+[sum(bool(re.match(r'^[+$−\-]?\d',str(row[i]))) for row in block['rows'])>
                             len(block['rows'])/2 for i in range(1,len(widths))]
            self.y-=3;self.rule(self.y,.8);self.y-=5
            for index,row in enumerate([block['headers']]+block['rows']):
                weight='bold' if index==0 else 'normal'
                texts=[self.lines(str(v),9.3,weight=weight,width=w-12) for v,w in zip(row,widths)]
                height=max(len(t) for t in texts)*11.4+7
                x=self.x
                for i,(lines,width) in enumerate(zip(texts,widths)):
                    for n,line in enumerate(lines):
                        left=x+5
                        if numeric[i]:
                            font=FontProperties(family='STIXGeneral',size=9.3,weight=weight)
                            extent=self.renderer.get_text_width_height_descent(line,font,ismath=False)[0]*72/self.fig.dpi
                            left=x+width-5-extent
                        self.text(left,self.y-2-n*11.4,line,9.3,weight=weight)
                    x+=width
                self.y-=height
                if index==0:self.rule(self.y,.5);self.y-=4
            self.rule(self.y,.8);self.y-=13

        def image(self,name,height):
            from scripts.build_unified_options_paper import FIG,CAPTIONS,NUMBERS
            im=Image.open(FIG/f'{name}.png')
            height=self.width*im.height/im.width
            y=self.y-height
            ax=self.fig.add_axes([self.x/612,y/792,self.width/612,height/792])
            ax.imshow(im);ax.axis('off');self.figure_axes.append(ax)
            self.figure_placements.append(dict(name=name,x=self.x,y=y,width=self.width,height=height))
            self.y-=height+5
            self.paragraph(f'Figure {NUMBERS[name]}. '+CAPTIONS[name],size=9,leading=11)

        def equation(self,latex):
            self.fig.text(.5,self.y/792,'$'+latex+'$',fontsize=11,color=INK,
                          family='STIXGeneral',ha='center',va='top',parse_math=True)
            self.y-=30

    return AcademicPage


def format_word(parts,word):
    """Apply the same restrained paper style to the editable Word export."""
    ns=word.NS
    tag=lambda name:'{'+ns['w']+'}'+name
    styles=word.ET.fromstring(parts['word/styles.xml'])
    sizes={'Normal':22,'Title':30,'Subtitle':22,'Heading2':24,'Equation':22,
           'Caption':18,'TableHeader':18,'TableText':18,'Reference':18}
    for style in styles.findall('w:style',ns):
        name=style.get(tag('styleId'))
        for font in style.findall('.//w:rFonts',ns):
            for key in ('ascii','hAnsi','eastAsia'):font.set(tag(key),'Times New Roman')
        for color in style.findall('.//w:color',ns):color.set(tag('val'),'111111')
        if name in sizes:
            for size in style.findall('.//w:sz',ns):size.set(tag('val'),str(sizes[name]))
        if name=='Normal':
            style.find('w:pPr/w:spacing',ns).set(tag('line'),'272')
        if name=='Equation':word.node(style.find('w:pPr',ns),'w:jc',{'w:val':'center'})
    for name,size,bold in [('PaperTitle',42,True),('PaperSubtitle',24,False),
                           ('PaperAuthor',23,False),('PaperMeta',20,False)]:
        style=word.node(styles,'w:style',{'w:type':'paragraph','w:styleId':name})
        word.node(style,'w:name',{'w:val':name});ppr=word.node(style,'w:pPr')
        word.node(ppr,'w:jc',{'w:val':'center'});word.node(ppr,'w:keepNext')
        word.node(ppr,'w:spacing',{'w:after':160,'w:line':240,'w:lineRule':'auto'})
        rpr=word.node(style,'w:rPr');word.node(rpr,'w:rFonts',{'w:ascii':'Times New Roman','w:hAnsi':'Times New Roman'})
        word.node(rpr,'w:sz',{'w:val':size});word.node(rpr,'w:color',{'w:val':'111111'})
        if bold:word.node(rpr,'w:b')
    document=word.ET.fromstring(parts['word/document.xml']);body=document.find('w:body',ns)
    body[0].find('w:pPr/w:pStyle',ns).set(tag('val'),'PaperTitle')
    old_subtitle=body[1];body.remove(old_subtitle)
    metadata=[(SUBTITLE,'PaperSubtitle'),(AUTHORS,'PaperAuthor'),(COURSE,'PaperMeta'),(DATE,'PaperMeta')]
    for index,(text,style) in enumerate([(text,style) for text,style in metadata if text],1):
        para=word.ET.Element(tag('p'));ppr=word.node(para,'w:pPr')
        word.node(ppr,'w:pStyle',{'w:val':style});word.node(word.node(para,'w:r'),'w:t',text=text)
        body.insert(index,para)
    abstract_seen=False
    for para in body.findall('w:p',ns):
        text=''.join(para.itertext())
        if text=='Abstract':
            word.node(para.find('w:pPr',ns),'w:jc',{'w:val':'center'});abstract_seen=True
        elif abstract_seen:
            word.node(para.find('w:pPr',ns),'w:ind',{'w:left':480,'w:right':480});abstract_seen=False
    for shading in list(document.findall('.//w:shd/..',ns)):
        for child in list(shading):
            if child.tag==tag('shd'):shading.remove(child)
    for table in document.findall('.//w:tbl',ns):
        borders=word.node(table.find('w:tblPr',ns),'w:tblBorders')
        for edge in ('top','bottom'):word.node(borders,'w:'+edge,{'w:val':'single','w:sz':6,'w:color':'111111'})
        for edge in ('left','right','insideH','insideV'):word.node(borders,'w:'+edge,{'w:val':'nil'})
        for cell in table.findall('w:tr',ns)[0].findall('w:tc',ns):
            border=word.node(cell.find('w:tcPr',ns),'w:tcBorders')
            word.node(border,'w:bottom',{'w:val':'single','w:sz':4,'w:color':'111111'})
    for node in document.findall('.//w:tblW',ns)+document.findall('.//w:gridCol',ns)+document.findall('.//w:tcW',ns):
        value=node.get(tag('w'))
        if value:node.set(tag('w'),str(round(int(value)*504/512)))
    for node in document.findall('.//w:color',ns):node.set(tag('val'),'111111')
    margin=document.find('w:body/w:sectPr/w:pgMar',ns)
    for key in ('top','bottom','left','right'):margin.set(tag(key),'1080')
    header=word.ET.fromstring(parts['word/header1.xml'])
    for child in list(header):header.remove(child)
    footer=word.ET.fromstring(parts['word/footer1.xml'])
    for child in list(footer):footer.remove(child)
    para=word.node(footer,'w:p');word.node(word.node(para,'w:pPr'),'w:jc',{'w:val':'center'})
    word.node(para,'w:fldSimple',{'w:instr':'PAGE'})
    for name,root in [('document',document),('styles',styles),('header1',header),('footer1',footer)]:
        parts['word/'+name+'.xml']=word.xml(root)
    return parts
