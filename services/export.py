import pandas as pd
import numpy as np
import io
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import json
import plotly.graph_objects as go
import plotly.io as pio

# Optional imports for PDF generation (install with: pip install reportlab)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class ExportManager:
    """
    Central manager for all export functionalities in the Process Capability Analysis Platform.
    """
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def export_raw_data(self, data: pd.DataFrame, filename_prefix: str = "process_data") -> bytes:
        """
        Export raw data to CSV format.
        
        Args:
            data: DataFrame containing the raw data
            filename_prefix: Prefix for the filename
            
        Returns:
            CSV data as bytes
        """
        output = io.StringIO()
        data.to_csv(output, index=False)
        return output.getvalue().encode('utf-8')
    
    def export_data_to_excel(self, data_dict: Dict[str, pd.DataFrame], filename_prefix: str = "process_analysis") -> bytes:
        """
        Export multiple datasets to Excel with different sheets.
        
        Args:
            data_dict: Dictionary with sheet names as keys and DataFrames as values
            filename_prefix: Prefix for the filename
            
        Returns:
            Excel data as bytes
        """
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return output.getvalue()
    
    def create_analysis_summary(
        self,
        data_series: pd.Series,
        capability_results: Dict[str, Any],
        tolerance_results: Optional[Dict[str, Any]] = None,
        specifications: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Create a comprehensive analysis summary table.
        
        Args:
            data_series: Original data series
            capability_results: Results from capability analysis (Normal, Empirical, Fitted)
            tolerance_results: Results from tolerance interval analysis
            specifications: LSL and USL values
            
        Returns:
            DataFrame with analysis summary
        """
        summary_data = []
        
        # Basic statistics
        summary_data.extend([
            ["Data Overview", "", ""],
            ["Sample Size", len(data_series), ""],
            ["Mean", f"{data_series.mean():.3f}", ""],
            ["Standard Deviation", f"{data_series.std():.3f}", ""],
            ["Minimum", f"{data_series.min():.3f}", ""],
            ["Maximum", f"{data_series.max():.3f}", ""],
            ["", "", ""],
        ])
        
        # Specifications
        if specifications:
            summary_data.extend([
                ["Specification Limits", "", ""],
                ["LSL (Lower Spec Limit)", specifications.get('LSL', 'Not Set'), ""],
                ["USL (Upper Spec Limit)", specifications.get('USL', 'Not Set'), ""],
                ["", "", ""],
            ])
        
        # Process Capability Results
        summary_data.append(["Process Capability Analysis", "", ""])
        
        for method, results in capability_results.items():
            if results and results.get('success'):
                ppk = results.get('Ppk', 'N/A')
                pp = results.get('Pp', 'N/A')
                ci_lower = results.get('Ppk_ci_lower', 'N/A')
                ci_upper = results.get('Ppk_ci_upper', 'N/A')
                
                summary_data.extend([
                    [f"{method} Method", "", ""],
                    ["  Ppk", f"{ppk:.3f}" if ppk != 'N/A' else 'N/A', ""],
                    ["  Pp", f"{pp:.3f}" if pp != 'N/A' else 'N/A', ""],
                    ["  95% CI", f"[{ci_lower:.3f}, {ci_upper:.3f}]" if ci_lower != 'N/A' else 'N/A', ""],
                ])
            else:
                summary_data.append([f"{method} Method", "Failed", ""])
        
        summary_data.append(["", "", ""])
        
        # Tolerance Intervals
        if tolerance_results and tolerance_results.get('success'):
            ti_result = tolerance_results.get('ti_result', {})
            if ti_result.get('success'):
                interval = ti_result.get('interval', (None, None))
                factor = ti_result.get('factor', 'N/A')
                n_lots = ti_result.get('n_lots', 'N/A')
                interval_type = ti_result.get('type', 'Unknown')
                
                summary_data.extend([
                    ["Tolerance Intervals (NoSlope Method)", "", ""],
                    ["  Type", interval_type.replace('-', ' ').title(), ""],
                    ["  Lower Bound", f"{interval[0]:.3f}" if interval[0] is not None else 'N/A', ""],
                    ["  Upper Bound", f"{interval[1]:.3f}" if interval[1] is not None else 'N/A', ""],
                    ["  Factor", f"{factor:.3f}" if factor != 'N/A' else 'N/A', ""],
                    ["  Number of Lots", str(n_lots), ""],
                ])
                
                # Coverage scan results
                scan_result = tolerance_results.get('scan_result', {})
                if scan_result.get('success'):
                    p_star = scan_result.get('p_star', 'N/A')
                    oos_ti = scan_result.get('oos_ti', 'N/A')
                    summary_data.extend([
                        ["  Max Coverage", f"{p_star*100:.1f}%" if p_star != 'N/A' else 'N/A', ""],
                        ["  Expected OOS", f"{oos_ti*100:.3f}%" if oos_ti != 'N/A' else 'N/A', ""],
                    ])
        
        # Create DataFrame
        df = pd.DataFrame(summary_data, columns=['Parameter', 'Value', 'Notes'])
        df['Analysis Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return df
    
    def generate_pdf_report(
        self,
        data_series: pd.Series,
        capability_results: Dict[str, Any],
        tolerance_results: Optional[Dict[str, Any]] = None,
        specifications: Optional[Dict[str, float]] = None,
        filename_prefix: str = "process_capability_report"
    ) -> bytes:
        """
        Generate a comprehensive PDF report.
        
        Args:
            data_series: Original data series
            capability_results: Results from capability analysis
            tolerance_results: Results from tolerance interval analysis
            specifications: LSL and USL values
            filename_prefix: Prefix for the filename
            
        Returns:
            PDF data as bytes
        """
        if not PDF_AVAILABLE:
            raise ImportError("PDF generation requires reportlab. Install with: pip install reportlab")
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Process Capability Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Metadata
        meta_data = [
            ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Sample Size:", str(len(data_series))],
            ["Analysis Platform:", "Process Capability Analysis Platform"]
        ]
        
        meta_table = Table(meta_data, colWidths=[2*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(meta_table)
        story.append(Spacer(1, 30))
        
        # Data Summary
        story.append(Paragraph("Data Summary", styles['Heading2']))
        
        data_summary = [
            ["Statistic", "Value"],
            ["Mean", f"{data_series.mean():.3f}"],
            ["Standard Deviation", f"{data_series.std():.3f}"],
            ["Minimum", f"{data_series.min():.3f}"],
            ["Maximum", f"{data_series.max():.3f}"],
            ["Range", f"{data_series.max() - data_series.min():.3f}"]
        ]
        
        if specifications:
            if specifications.get('LSL') is not None:
                data_summary.append(["LSL", f"{specifications['LSL']:.3f}"])
            if specifications.get('USL') is not None:
                data_summary.append(["USL", f"{specifications['USL']:.3f}"])
        
        data_table = Table(data_summary, colWidths=[2*inch, 2*inch])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(data_table)
        story.append(Spacer(1, 30))
        
        # Process Capability Results
        story.append(Paragraph("Process Capability Analysis", styles['Heading2']))
        
        cap_data = [["Method", "Ppk", "Pp", "95% CI Lower", "95% CI Upper", "Status"]]
        
        for method, results in capability_results.items():
            if results and results.get('success'):
                ppk = results.get('Ppk', 'N/A')
                pp = results.get('Pp', 'N/A')
                ci_lower = results.get('Ppk_ci_lower', 'N/A')
                ci_upper = results.get('Ppk_ci_upper', 'N/A')
                
                # Determine status
                if ppk != 'N/A':
                    if ppk >= 1.33:
                        status = "Capable"
                    elif ppk >= 1.0:
                        status = "Marginal"
                    else:
                        status = "Incapable"
                else:
                    status = "Unknown"
                
                cap_data.append([
                    method,
                    f"{ppk:.3f}" if ppk != 'N/A' else 'N/A',
                    f"{pp:.3f}" if pp != 'N/A' else 'N/A',
                    f"{ci_lower:.3f}" if ci_lower != 'N/A' else 'N/A',
                    f"{ci_upper:.3f}" if ci_upper != 'N/A' else 'N/A',
                    status
                ])
            else:
                cap_data.append([method, "Failed", "Failed", "Failed", "Failed", "Failed"])
        
        cap_table = Table(cap_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch])
        cap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(cap_table)
        story.append(Spacer(1, 30))
        
        # Tolerance Intervals (if available)
        if tolerance_results and tolerance_results.get('success'):
            ti_result = tolerance_results.get('ti_result', {})
            if ti_result.get('success'):
                story.append(Paragraph("Tolerance Intervals Analysis", styles['Heading2']))
                
                interval = ti_result.get('interval', (None, None))
                factor = ti_result.get('factor', 'N/A')
                n_lots = ti_result.get('n_lots', 'N/A')
                interval_type = ti_result.get('type', 'Unknown')
                
                ti_data = [
                    ["Parameter", "Value"],
                    ["Method", "NoSlope (Random Effects)"],
                    ["Interval Type", interval_type.replace('-', ' ').title()],
                    ["Lower Bound", f"{interval[0]:.3f}" if interval[0] is not None else 'N/A'],
                    ["Upper Bound", f"{interval[1]:.3f}" if interval[1] is not None else 'N/A'],
                    ["Factor", f"{factor:.3f}" if factor != 'N/A' else 'N/A'],
                    ["Number of Lots", str(n_lots)]
                ]
                
                # Add coverage scan results if available
                scan_result = tolerance_results.get('scan_result', {})
                if scan_result.get('success'):
                    p_star = scan_result.get('p_star', 'N/A')
                    oos_ti = scan_result.get('oos_ti', 'N/A')
                    ti_data.extend([
                        ["Max Coverage", f"{p_star*100:.1f}%" if p_star != 'N/A' else 'N/A'],
                        ["Expected OOS", f"{oos_ti*100:.3f}%" if oos_ti != 'N/A' else 'N/A']
                    ])
                
                ti_table = Table(ti_data, colWidths=[2*inch, 2*inch])
                ti_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(ti_table)
        
        # Footer
        story.append(Spacer(1, 50))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Generated by Process Capability Analysis Platform", footer_style))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def export_chart_as_image(self, fig: go.Figure, format: str = "png", width: int = 1200, height: int = 800) -> bytes:
        """
        Export a Plotly figure as image bytes.
        
        Args:
            fig: Plotly figure object
            format: Image format ('png', 'svg', 'pdf', 'html')
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Image data as bytes
        """
        if format.lower() == 'html':
            return pio.to_html(fig, include_plotlyjs=True).encode('utf-8')
        else:
            return pio.to_image(fig, format=format, width=width, height=height)
    
    def create_filename(self, prefix: str, extension: str) -> str:
        """
        Create a standardized filename with timestamp.
        
        Args:
            prefix: Filename prefix
            extension: File extension (with or without dot)
            
        Returns:
            Formatted filename
        """
        if not extension.startswith('.'):
            extension = '.' + extension
        return f"{prefix}_{self.timestamp}{extension}"