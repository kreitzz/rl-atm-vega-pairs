"""Export only the teammate's original title and six paper-discussion slides."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import time

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'slides/teammate_only'
COMMIT = '8d3aa74'


def prepare():
    original = subprocess.check_output(['git', 'show', f'{COMMIT}:slides/index.html'], cwd=ROOT).decode()
    sections = re.findall(r'<section\b.*?</section>', original, flags=re.S)
    document = original
    for section in sections[7:]:
        document = document.replace(section, '', 1)
    start = document.index('  // Ours: combined-basket Sharpe')
    end = document.index('</script>', start)
    document = document[:start] + document[end:]
    assert re.findall(r'<section\b.*?</section>', document, flags=re.S) == sections[:7]
    # Embed the original image bytes so the exported HTML is uploadable alone.
    for name in ['paper_table3.png', 'paper_table4.png']:
        payload = base64.b64encode((ROOT / 'slides/img' / name).read_bytes()).decode()
        document = document.replace(f'src="img/{name}"', f'src="data:image/png;base64,{payload}"')
    export_css = '''<style>
      @page {size:13.333333in 7.5in;margin:0}
      html,body,.deck {width:1280px;margin:0;padding:0}
      .slide {width:1280px;height:720px;min-height:0;padding:40px 64px 40px;
              box-sizing:border-box;break-inside:avoid;page-break-after:always;break-after:page}
      .slide:last-child {page-break-after:auto;break-after:auto}
      .slide h1 {font-size:34px}.slide h2 {font-size:18px;margin-bottom:18px}
      .slide .body {gap:14px}.slide .two-col {gap:30px}
      .slide img {max-height:410px;object-fit:contain}
      .title-slide h1 {font-size:46px}.title-slide h2 {font-size:22px}
      .progress,.scrollhint {display:none}.pageno {bottom:10px}
      @media print {.pageno {display:none}}
    </style>'''
    document = document.replace('</head>', export_css + '\n</head>', 1)
    document = document.replace('</body>', '''<script>
      addEventListener('load',()=>setTimeout(()=>{
        document.body.setAttribute('data-layout-overflow',JSON.stringify(
          [...document.querySelectorAll('.slide')].map((s,i)=>({slide:i+1,
          excess:Math.max(0,s.scrollHeight-s.clientHeight)})).filter(x=>x.excess>2)));
      },100));
    </script></body>''', 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'teammate_paper_slides.html').write_text(document)
    notes = subprocess.check_output(['git', 'show', f'{COMMIT}:slides/script.md'], cwd=ROOT)
    (OUT / 'speaker_script.md').write_bytes(notes)
    (OUT / 'export_audit.json').write_text(json.dumps(dict(
        original_commit=COMMIT, exported_original_slides=list(range(1,8)),
        original_sections_preserved_before_asset_embedding=True,
        original_script_sha256=hashlib.sha256(notes).hexdigest(),
        original_images_embedded=True, our_research_included=False), indent=2))
    return OUT / 'teammate_paper_slides.html'


def cpu_seconds(value):
    parts = value.split(':')
    return sum(float(part) * 60**index for index, part in enumerate(reversed(parts)))


def render(source):
    browser = Path.home() / 'Library/Caches/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-mac-arm64/chrome-headless-shell'
    assert browser.is_file()
    destination = OUT / 'teammate_paper_slides.pdf'
    command = [str(browser), '--headless', '--disable-gpu', '--single-process',
        '--no-zygote', '--renderer-process-limit=1', '--num-raster-threads=1',
        '--disable-background-networking', '--disable-breakpad', '--disable-crash-reporter',
        '--disable-threaded-animation', '--disable-threaded-scrolling',
        '--disable-extensions', '--no-pdf-header-footer', '--hide-scrollbars',
        '--window-size=1280,720', '--virtual-time-budget=1000', '--dump-dom',
        f'--user-data-dir={OUT / "browser_profile"}',
        f'--print-to-pdf={destination}', source.as_uri()]
    log = OUT / 'render_log.txt'
    started = time.monotonic()
    peak_rss = 0
    # A single browser process. Count CPU across all its threads, reserving
    # 5% of a core for the supervising Python process under the outer guard.
    with log.open('w') as stdout, (OUT / 'render_stderr.txt').open('w') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        try:
            while child.poll() is None:
                if time.monotonic()-started > 90:
                    raise TimeoutError('PDF export exceeded 90 seconds')
                snapshot = subprocess.run(['ps', '-o', 'rss=,cputime=', '-p', str(child.pid)],
                    capture_output=True, text=True)
                if snapshot.stdout.strip():
                    rss, cpu = snapshot.stdout.split()
                    peak_rss = max(peak_rss, int(rss)*1024)
                    if peak_rss > 3*1024**3:
                        raise MemoryError('Browser exceeded 3 GiB')
                    pause = cpu_seconds(cpu)/.45 - (time.monotonic()-started)
                    if pause > 0:
                        child.send_signal(signal.SIGSTOP)
                        time.sleep(min(pause, .2))
                        child.send_signal(signal.SIGCONT)
                time.sleep(.02)
            assert child.returncode == 0, (OUT / 'render_stderr.txt').read_text()[-1200:]
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGCONT)
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    dom = log.read_text()
    overflow = re.search(r'data-layout-overflow="([^"]*)"', dom)
    assert overflow is not None and overflow.group(1) == '[]', overflow.group(1) if overflow else 'Missing layout check'
    pdf = PdfReader(destination)
    assert len(pdf.pages) == 7, len(pdf.pages)
    assert 'EMRT' in pdf.pages[2].extract_text()
    assert 'Q-learning' in pdf.pages[4].extract_text()
    audit = json.loads((OUT / 'export_audit.json').read_text())
    audit.update(pdf_pages=len(pdf.pages), layout_overflow=False, browser_peak_rss_bytes=peak_rss,
                 cpu_limit_fraction_of_one_core=.5, numerical_workers=1)
    (OUT / 'export_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()
    source = prepare()
    if args.pdf:
        render(source)
    else:
        print(str(source), flush=True)


if __name__ == '__main__':
    main()
