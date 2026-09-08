#!/usr/bin/env python3
"""Bundle index.html + vendor/ into one self-contained dist/index.html."""
import base64, os, re, pathlib

root = pathlib.Path(__file__).parent
html = (root / "index.html").read_text(encoding="utf-8")

# fonts.css -> inline <style> with woff2 as data URIs
link = re.search(r'<link href="vendor/([^"]+\.css)" rel="stylesheet">', html).group(0)
css_name = re.search(r'vendor/([^"]+\.css)', link).group(1)
css = (root / "vendor" / css_name).read_text(encoding="utf-8")
def font_uri(m):
    data = (root / "vendor" / m.group(1)).read_bytes()
    return "url(data:font/woff2;base64," + base64.b64encode(data).decode() + ")"
css = re.sub(r"url\((fonts/[^)]+)\)", font_uri, css)
html = html.replace(link, "<style>\n" + css + "\n</style>", 1)

# scripts -> inline
for name in ("reveal.min.js", "notes.min.js", "chart.umd.js"):
    js = (root / "vendor" / name).read_text(encoding="utf-8")
    assert "</script" not in js, name
    tag = f'<script src="vendor/{name}"></script>'
    assert html.count(tag) == 1, tag
    html = html.replace(tag, "<script>\n" + js + "\n</script>", 1)

assert "vendor/" not in html, "unresolved vendor reference"
out = root / "dist" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1024/1024:.2f} MB)")
