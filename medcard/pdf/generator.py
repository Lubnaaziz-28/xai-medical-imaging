"""PDF generation for model cards using Jinja2 templates and WeasyPrint."""

from pathlib import Path
from typing import Optional, Dict, Any, List

import jinja2

from medcard.pdf.template import TEMPLATE_HTML, TEMPLATE_CSS


class ModelCardPDFGenerator:
    """Generate PDF model cards from HTML templates using WeasyPrint."""

    def __init__(
        self,
        template_html: Optional[str] = None,
        template_css: Optional[str] = None,
        output_dir: str = "output",
    ):
        """
        Initialize the PDF generator.

        Args:
            template_html: Jinja2 HTML template (uses default if None)
            template_css: CSS stylesheet (uses default if None)
            output_dir: Directory for generated artifacts
        """
        self.template_html = template_html or TEMPLATE_HTML
        self.template_css = template_css or TEMPLATE_CSS
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _render_html(self, context: Dict[str, Any]) -> str:
        """Render the HTML template with the given context."""
        template = jinja2.Template(self.template_html)
        # Inject CSS into context
        context = dict(context)
        context.setdefault("css", self.template_css)
        return template.render(**context)

    def generate(
        self,
        context: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a PDF model card.

        Args:
            context: Dictionary with template variables (model_name, metrics, etc.)
            output_path: Output PDF path (default: output_dir/model_card.pdf)

        Returns:
            Path to the generated PDF file
        """
        html_content = self._render_html(context)

        if output_path is None:
            output_path = str(self.output_dir / "model_card.pdf")
        else:
            output_path = str(output_path)

        try:
            from weasyprint import HTML
        except ImportError as e:
            raise ImportError(
                "WeasyPrint is required for PDF generation. "
                "Install it with: pip install weasyprint"
            ) from e

        HTML(string=html_content, base_url=str(self.output_dir)).write_pdf(output_path)
        return output_path

    def generate_from_dict(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a PDF from a flat dictionary of model card data.

        Maps common keys to template variables with sensible defaults.
        """
        context = self._build_context(data)
        return self.generate(context, output_path)

    def _build_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build template context from a flat data dictionary."""
        context: Dict[str, Any] = {
            "model_name": data.get("model_name", "Medical Imaging Model"),
            "model_version": data.get("model_version", "1.0.0"),
            "author": data.get("author", "Lubna Research Team"),
            "author_email": data.get("author_email", "research@lubna.ai"),
            "date": data.get("date", ""),
            "intended_use": data.get("intended_use", ""),
            "target_population": data.get("target_population", ""),
            "clinical_context": data.get("clinical_context", ""),
            "limitations": data.get("limitations", ""),
            "ethical_considerations": data.get("ethical_considerations", ""),
            "include_cover_page": data.get("include_cover_page", True),
            "include_appendix": data.get("include_appendix", True),
            "generate_xai": data.get("generate_xai", False),
            "generate_fairness": data.get("generate_fairness", False),
            "metrics": data.get("metrics", []),
            "xai_visualizations": data.get("xai_visualizations", []),
            "fairness": data.get("fairness", {}),
        }
        return context

    def save_html(self, context: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Save rendered HTML to a file (useful for debugging).

        Returns:
            Path to the HTML file
        """
        html_content = self._render_html(context)
        if output_path is None:
            output_path = str(self.output_dir / "model_card.html")
        Path(output_path).write_text(html_content, encoding="utf-8")
        return output_path