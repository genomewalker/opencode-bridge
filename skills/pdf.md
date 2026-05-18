# PDF Reading

Read and extract text from PDF files using the `pdf_read` tool.

## Usage

```
pdf_read(path="/absolute/path/to/file.pdf")              # first 30 pages
pdf_read(path="...", pages="info")                        # metadata only
pdf_read(path="...", pages="1-10")                        # page range
pdf_read(path="...", pages="5")                           # single page
pdf_read(path="...", pages="all", max_pages=50)           # up to 50 pages
pdf_read(path="...", pages="1-20", ingest=true)           # extract + store in memory
```

## Strategy for large PDFs

1. Call `pdf_read(pages="info")` to get page count and metadata.
2. Read in chunks of 20-30 pages at a time.
3. Use `ingest=true` on the first full pass to store content in chitta for later recall.
4. Use `recall(query="...")` in subsequent questions instead of re-reading.

## Capabilities

- **Text extraction** with layout preservation (pdfplumber primary, pypdf fallback)
- **Table detection** — tables rendered as pipe-delimited rows under `[table]` header
- **Metadata** — title, author, subject, creator, page count
- **Chitta ingestion** — auto-stores extracted text as searchable memories

## Libraries (installed)

| Library | Role |
|---------|------|
| pdfplumber | Primary — layout-aware text + table extraction |
| pypdf | Fallback — pure Python, reliable for text-only PDFs |
| pymupdf | Available via `import pymupdf` for advanced ops (images, annotations) |

## When pdfplumber is best

- PDFs with columns, tables, or complex layouts
- Academic papers, financial reports, data sheets

## When to use pypdf directly (via bash tool)

```python
from pypdf import PdfReader, PdfWriter
# merge, split, rotate, encrypt — manipulation operations
```

## OCR for scanned PDFs

If `pdf_read` returns empty pages, the PDF is likely scanned (image-only).
Use `bash` tool with pytesseract + pdf2image (install separately if needed):

```bash
python3 -c "
from pdf2image import convert_from_path
import pytesseract
pages = convert_from_path('file.pdf', first_page=1, last_page=5)
for i, img in enumerate(pages):
    print(f'--- Page {i+1} ---')
    print(pytesseract.image_to_string(img))
"
```
