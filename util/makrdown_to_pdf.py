import os
import tempfile
import markdown2
import logging
from weasyprint import HTML
from .errors import PDFGenerationError

logger = logging.getLogger(__name__)

def markdown_to_pdf(md: str) -> str:
    """
    Converts Markdown to PDF and returns the generated PDF path.
    """
    logger.info("Inside markdown_to_pdf function")

    pdf_path = None

    try:
        if not md or not md.strip():
            raise PDFGenerationError(
                "Markdown content is empty"
            )

        logger.info("Converting Markdown to HTML")

        html = markdown2.markdown(
            md,
            extras=["tables", "fenced-code-blocks"],
        )

        if not html.strip():
            raise PDFGenerationError(
                "Markdown conversion produced empty HTML"
            )

        logger.info("Converting HTML to PDF")

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as pdf_file:
            pdf_path = pdf_file.name

        HTML(string=html).write_pdf(pdf_path)

        if os.path.getsize(pdf_path) == 0:
            raise PDFGenerationError(
                "Generated PDF is empty"
            )

        logger.info("PDF conversion complete")

        return pdf_path

    except PDFGenerationError:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise

    except Exception as e:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)

        raise PDFGenerationError(
            "PDF generation failed"
        ) from e
