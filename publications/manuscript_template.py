"""
Manuscript Preparation Tools for Bayesian PDE Research

This module provides automated tools for generating publication-ready manuscripts,
including LaTeX templates, figure generation, reference management, and result
compilation for research in Bayesian PDE inverse problems.

Features:
- LaTeX manuscript templates with academic journal formatting
- Automated result table generation from experimental data
- Citation management and bibliography integration
- Figure compilation and placement optimization
- Supplementary material generation
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import tempfile
import shutil


class ManuscriptGenerator:
    """
    Automated manuscript generation for Bayesian PDE research publications.
    
    Generates LaTeX manuscripts with proper academic formatting, automated
    result integration, and citation management for submission to journals
    like Journal of Computational Physics, SIAM journals, etc.
    """
    
    def __init__(self, output_dir: str = "manuscripts", template_style: str = "jcp"):
        """
        Initialize manuscript generator.
        
        Parameters:
        -----------
        output_dir : str
            Directory for manuscript output
        template_style : str
            Journal template style ('jcp', 'siam', 'ieee', 'elsevier')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.template_style = template_style
        
        # Journal-specific configurations
        self.journal_configs = {
            'jcp': {
                'documentclass': 'article',
                'packages': ['amsmath', 'amssymb', 'graphicx', 'natbib', 'algorithm2e'],
                'citation_style': 'unsrtnat',
                'line_spacing': '1.5',
                'font_size': '12pt'
            },
            'siam': {
                'documentclass': 'siamltex',
                'packages': ['amsmath', 'amssymb', 'graphicx', 'natbib'],
                'citation_style': 'siamplain',
                'line_spacing': '1.0',
                'font_size': '11pt'
            },
            'ieee': {
                'documentclass': 'IEEEtran',
                'packages': ['amsmath', 'amssymb', 'graphicx', 'cite'],
                'citation_style': 'IEEEtran',
                'line_spacing': '1.0',
                'font_size': '10pt'
            }
        }
    
    def create_manuscript_template(self, title: str, authors: List[Dict], 
                                 abstract: str = "", keywords: List[str] = None) -> str:
        """
        Create LaTeX manuscript template with proper academic formatting.
        
        Parameters:
        -----------
        title : str
            Manuscript title
        authors : List[Dict]
            Author information with affiliations
        abstract : str
            Manuscript abstract
        keywords : List[str]
            Keywords for the manuscript
            
        Returns:
        --------
        str
            LaTeX template content
        """
        config = self.journal_configs[self.template_style]
        keywords = keywords or []
        
        # Generate author string
        author_lines = []
        for i, author in enumerate(authors):
            name = author.get('name', 'Author Name')
            affiliation = author.get('affiliation', 'Institution')
            email = author.get('email', '')
            
            if self.template_style == 'jcp':
                author_lines.append(f"\\author{{{name}}}\\thanks{{{affiliation}. Email: {email}}}")
            else:
                author_lines.append(f"\\author{{{name}\\\\{affiliation}\\\\{email}}}")
        
        author_block = '\n'.join(author_lines)
        
        # Keywords formatting
        keywords_str = ', '.join(keywords) if keywords else "Bayesian methods, inverse problems, uncertainty quantification"
        
        template = f"""\\documentclass[{config['font_size']}]{{article}}

% Package imports
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{natbib}}
\\usepackage{{algorithm2e}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}
\\usepackage{{hyperref}}
\\usepackage{{cleveref}}
\\usepackage{{lineno}}
\\usepackage{{setspace}}

% Line spacing
\\{config['line_spacing']}spacing

% Line numbers for review
\\linenumbers

% Hyperref setup
\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    citecolor=red
}}

% Math shortcuts
\\newcommand{{\\R}}{{\\mathbb{{R}}}}
\\newcommand{{\\E}}{{\\mathbb{{E}}}}
\\newcommand{{\\Prob}}{{\\mathbb{{P}}}}
\\newcommand{{\\norm}}[1]{{\\|#1\\|}}
\\newcommand{{\\inner}}[2]{{\\langle #1, #2 \\rangle}}

\\title{{{title}}}

{author_block}

\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{abstract if abstract else "Abstract content will be inserted here. This work presents a comprehensive framework for Bayesian inference in partial differential equation inverse problems with certified uncertainty quantification..."}
\\end{{abstract}}

\\textbf{{Keywords:}} {keywords_str}

\\section{{Introduction}}

% Introduction content will be generated automatically

\\section{{Mathematical Framework}}

\\subsection{{Problem Formulation}}

Consider the parameter-dependent partial differential equation:
\\begin{{align}}
L(\\theta) u &= f \\quad \\text{{in }} \\Omega \\\\
B(\\theta) u &= g \\quad \\text{{on }} \\partial\\Omega
\\end{{align}}
where $\\theta \\in \\Theta \\subset \\R^d$ are unknown parameters to be estimated from noisy observations.

\\subsection{{Bayesian Framework}}

% Mathematical framework content

\\section{{Theoretical Results}}

% Theoretical contributions will be inserted here

\\section{{Computational Methods}}

% Implementation details

\\section{{Numerical Experiments}}

% Experimental results will be inserted here

\\section{{Real-World Applications}}

% Application case studies

\\section{{Conclusions}}

% Conclusions and future work

\\section*{{Acknowledgments}}

The authors thank the reviewers for their constructive feedback. This work was supported by NSF Grant DMS-XXXXXX.

\\bibliographystyle{{{config['citation_style']}}}
\\bibliography{{references}}

\\end{{document}}
"""
        return template
    
    def generate_results_table(self, results_data: Dict[str, Any], 
                             caption: str = "Experimental Results") -> str:
        """
        Generate LaTeX table from experimental results.
        
        Parameters:
        -----------
        results_data : Dict
            Experimental results data
        caption : str
            Table caption
            
        Returns:
        --------
        str
            LaTeX table code
        """
        if isinstance(results_data, dict) and 'methods' in results_data:
            # Benchmark comparison format
            methods = results_data['methods']
            metrics = results_data.get('metrics', ['MSE', 'Time', 'Coverage'])
            
            # Create table header
            header = "Method & " + " & ".join(metrics) + " \\\\\n"
            
            # Create table rows
            rows = []
            for method, values in methods.items():
                row_values = []
                for metric in metrics:
                    if metric.lower() in values:
                        value = values[metric.lower()]
                        if isinstance(value, (int, float)):
                            if metric.lower() == 'time':
                                row_values.append(f"{value:.1f}")
                            elif metric.lower() == 'coverage':
                                row_values.append(f"{value:.1f}\\%")
                            else:
                                row_values.append(f"{value:.4f}")
                        else:
                            row_values.append(str(value))
                    else:
                        row_values.append("--")
                
                rows.append(f"{method} & " + " & ".join(row_values) + " \\\\")
            
            table_content = "\n".join(rows)
            
        else:
            # DataFrame format
            if isinstance(results_data, pd.DataFrame):
                df = results_data
            else:
                df = pd.DataFrame(results_data)
            
            # Format numerical columns
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "--")
            
            header = " & ".join(df.columns) + " \\\\\n"
            rows = []
            for _, row in df.iterrows():
                rows.append(" & ".join(str(val) for val in row.values) + " \\\\")
            
            table_content = "\n".join(rows)
        
        table_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{{'c' * (len(metrics) + 1 if 'methods' in results_data else len(df.columns))}}}
\\toprule
{header}
\\midrule
{table_content}
\\bottomrule
\\end{{tabular}}
\\label{{tab:results}}
\\end{{table}}"""
        
        return table_latex
    
    def create_bibliography_file(self, references: List[Dict[str, str]]) -> str:
        """
        Create BibTeX bibliography file.
        
        Parameters:
        -----------
        references : List[Dict]
            List of reference dictionaries
            
        Returns:
        --------
        str
            BibTeX content
        """
        bib_entries = []
        
        # Add default references for Bayesian PDE methods
        default_refs = [
            {
                'type': 'article',
                'key': 'stuart2010inverse',
                'title': 'Inverse problems: a Bayesian perspective',
                'author': 'Stuart, Andrew M',
                'journal': 'Acta numerica',
                'volume': '19',
                'pages': '451--559',
                'year': '2010',
                'publisher': 'Cambridge University Press'
            },
            {
                'type': 'book',
                'key': 'kaipio2004statistical',
                'title': 'Statistical and computational inverse problems',
                'author': 'Kaipio, Jari and Somersalo, Erkki',
                'volume': '160',
                'year': '2004',
                'publisher': 'Springer Science \\& Business Media'
            },
            {
                'type': 'article',
                'key': 'dashti2013bayesian',
                'title': 'The Bayesian approach to inverse problems',
                'author': 'Dashti, Masoumeh and Stuart, Andrew M',
                'journal': 'Handbook of uncertainty quantification',
                'pages': '311--428',
                'year': '2017',
                'publisher': 'Springer'
            }
        ]
        
        all_refs = default_refs + references
        
        for ref in all_refs:
            entry_type = ref.get('type', 'article')
            key = ref.get('key', 'unknown')
            
            bib_entry = f"@{entry_type}{{{key},\n"
            
            for field, value in ref.items():
                if field not in ['type', 'key']:
                    bib_entry += f"  {field} = {{{value}}},\n"
            
            bib_entry = bib_entry.rstrip(',\n') + "\n}\n\n"
            bib_entries.append(bib_entry)
        
        return ''.join(bib_entries)
    
    def compile_manuscript(self, tex_file: str, output_name: str = None) -> bool:
        """
        Compile LaTeX manuscript to PDF.
        
        Parameters:
        -----------
        tex_file : str
            Path to LaTeX file
        output_name : str, optional
            Output PDF name
            
        Returns:
        --------
        bool
            True if compilation successful
        """
        tex_path = Path(tex_file)
        if not tex_path.exists():
            print(f"LaTeX file {tex_file} not found")
            return False
        
        # Change to manuscript directory for compilation
        original_dir = os.getcwd()
        os.chdir(tex_path.parent)
        
        try:
            # Run pdflatex twice for cross-references
            for _ in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_path.name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"LaTeX compilation failed:\n{result.stderr}")
                    return False
            
            # Run bibtex for bibliography
            subprocess.run(['bibtex', tex_path.stem], capture_output=True)
            
            # Final pdflatex run
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_path.name],
                capture_output=True,
                text=True
            )
            
            # Rename output if requested
            if output_name and result.returncode == 0:
                pdf_file = tex_path.with_suffix('.pdf')
                if pdf_file.exists():
                    new_pdf = tex_path.parent / f"{output_name}.pdf"
                    shutil.move(str(pdf_file), str(new_pdf))
            
            return result.returncode == 0
            
        except FileNotFoundError:
            print("pdflatex not found. Please install LaTeX distribution.")
            return False
        
        finally:
            os.chdir(original_dir)
    
    def generate_complete_manuscript(self, config: Dict[str, Any], 
                                   results_data: Dict[str, Any] = None) -> str:
        """
        Generate complete manuscript with all sections.
        
        Parameters:
        -----------
        config : Dict
            Manuscript configuration
        results_data : Dict, optional
            Experimental results to include
            
        Returns:
        --------
        str
            Path to generated manuscript
        """
        # Extract configuration
        title = config.get('title', 'Bayesian PDE Inverse Problems with Certified Uncertainty')
        authors = config.get('authors', [{'name': 'Author Name', 'affiliation': 'Institution'}])
        abstract = config.get('abstract', '')
        keywords = config.get('keywords', [])
        
        # Generate base template
        template = self.create_manuscript_template(title, authors, abstract, keywords)
        
        # Add results table if provided
        if results_data:
            results_table = self.generate_results_table(results_data)
            # Insert table before conclusions
            template = template.replace(
                '\\section{Conclusions}',
                f'{results_table}\n\n\\section{{Conclusions}}'
            )
        
        # Write manuscript file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manuscript_{timestamp}.tex"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(template)
        
        # Generate bibliography
        references = config.get('references', [])
        bib_content = self.create_bibliography_file(references)
        bib_path = filepath.parent / "references.bib"
        
        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write(bib_content)
        
        print(f"Manuscript generated: {filepath}")
        print(f"Bibliography generated: {bib_path}")
        
        return str(filepath)


class SupplementaryMaterialGenerator:
    """
    Generate supplementary materials for manuscript submission.
    
    Creates additional figures, detailed algorithm descriptions,
    extended experimental results, and code documentation.
    """
    
    def __init__(self, output_dir: str = "supplementary"):
        """Initialize supplementary material generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_algorithm_appendix(self, algorithms: List[Dict]) -> str:
        """
        Generate detailed algorithm descriptions.
        
        Parameters:
        -----------
        algorithms : List[Dict]
            Algorithm specifications
            
        Returns:
        --------
        str
            LaTeX algorithm appendix
        """
        appendix_content = [
            "\\section*{Appendix A: Detailed Algorithms}",
            ""
        ]
        
        for i, algo in enumerate(algorithms, 1):
            name = algo.get('name', f'Algorithm {i}')
            description = algo.get('description', '')
            steps = algo.get('steps', [])
            
            algo_latex = f"""
\\subsection*{{A.{i} {name}}}

{description}

\\begin{{algorithm}}[H]
\\SetAlgoLined
\\KwData{{{algo.get('input', 'Input data')}}}
\\KwResult{{{algo.get('output', 'Output result')}}}
"""
            
            for j, step in enumerate(steps, 1):
                algo_latex += f"\\tcp{{{step}}}\n"
            
            algo_latex += f"\\caption{{{name}}}\n\\end{{algorithm}}\n"
            
            appendix_content.append(algo_latex)
        
        return '\n'.join(appendix_content)
    
    def generate_extended_results(self, detailed_results: Dict) -> str:
        """
        Generate extended experimental results section.
        
        Parameters:
        -----------
        detailed_results : Dict
            Detailed experimental data
            
        Returns:
        --------
        str
            LaTeX extended results section
        """
        content = [
            "\\section*{Appendix B: Extended Experimental Results}",
            ""
        ]
        
        # Add convergence analysis
        if 'convergence' in detailed_results:
            content.extend([
                "\\subsection*{B.1 Convergence Analysis}",
                "Detailed convergence analysis for all test cases...",
                ""
            ])
        
        # Add parameter sensitivity
        if 'sensitivity' in detailed_results:
            content.extend([
                "\\subsection*{B.2 Parameter Sensitivity Analysis}",
                "Analysis of algorithm performance across parameter ranges...",
                ""
            ])
        
        # Add computational complexity
        if 'complexity' in detailed_results:
            content.extend([
                "\\subsection*{B.3 Computational Complexity}",
                "Detailed analysis of computational requirements...",
                ""
            ])
        
        return '\n'.join(content)
    
    def create_supplementary_document(self, content_sections: List[str]) -> str:
        """
        Create complete supplementary material document.
        
        Parameters:
        -----------
        content_sections : List[str]
            LaTeX content sections
            
        Returns:
        --------
        str
            Path to supplementary document
        """
        header = """\\documentclass[11pt]{article}
\\usepackage{amsmath,amssymb,amsfonts}
\\usepackage{graphicx}
\\usepackage{algorithm2e}
\\usepackage{booktabs}
\\usepackage{hyperref}

\\title{Supplementary Material: Bayesian PDE Inverse Problems}
\\author{Authors}
\\date{\\today}

\\begin{document}
\\maketitle

"""
        
        footer = "\\end{document}"
        
        full_content = header + '\n'.join(content_sections) + '\n' + footer
        
        # Write supplementary document
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"supplementary_{timestamp}.tex"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return str(filepath)


def demo_manuscript_generation():
    """Demonstrate manuscript generation capabilities."""
    print("Manuscript Preparation Tools Demo")
    print("=" * 40)
    
    # Initialize generator
    generator = ManuscriptGenerator(template_style='jcp')
    
    # Configuration
    config = {
        'title': 'Certified Uncertainty Quantification for Bayesian PDE Inverse Problems',
        'authors': [
            {
                'name': 'Jane Researcher',
                'affiliation': 'Department of Mathematics, University of Excellence',
                'email': 'jane.researcher@university.edu'
            },
            {
                'name': 'John Collaborator',
                'affiliation': 'Institute for Computational Science, Tech University',
                'email': 'john.collaborator@tech.edu'
            }
        ],
        'abstract': '''We present a comprehensive framework for Bayesian inverse problems in partial differential equations with rigorous uncertainty quantification guarantees. Our approach combines adaptive concentration inequalities, minimax optimal posterior contraction rates, and sharp PAC-Bayes bounds to provide certified uncertainty intervals with explicit finite-sample constants.''',
        'keywords': ['Bayesian methods', 'inverse problems', 'uncertainty quantification', 'PDEs'],
        'references': [
            {
                'type': 'article',
                'key': 'author2024method',
                'title': 'New Method for Uncertainty Quantification',
                'author': 'Author, A. and Collaborator, B.',
                'journal': 'Journal of Computational Methods',
                'year': '2024',
                'volume': '15',
                'pages': '123--145'
            }
        ]
    }
    
    # Sample results data
    results_data = {
        'methods': {
            'Our Method': {'mse': 0.0045, 'time': 12.3, 'coverage': 94.2},
            'Tikhonov': {'mse': 0.0067, 'time': 2.1, 'coverage': 78.4},
            'EnKF': {'mse': 0.0052, 'time': 8.7, 'coverage': 87.1},
            'MCMC': {'mse': 0.0048, 'time': 45.6, 'coverage': 92.8}
        },
        'metrics': ['MSE', 'Time', 'Coverage']
    }
    
    # Generate manuscript
    manuscript_path = generator.generate_complete_manuscript(config, results_data)
    print(f"Generated manuscript: {manuscript_path}")
    
    # Generate supplementary materials
    supp_generator = SupplementaryMaterialGenerator()
    
    algorithms = [
        {
            'name': 'Adaptive MCMC Sampler',
            'description': 'Adaptive Metropolis-Hastings sampler with concentration bounds.',
            'input': 'Prior distribution, likelihood function, observations',
            'output': 'Posterior samples with uncertainty certificates',
            'steps': [
                'Initialize chain at prior mode',
                'Adapt proposal covariance using sample history',
                'Compute concentration bounds at each iteration',
                'Update uncertainty certificates'
            ]
        }
    ]
    
    algo_appendix = supp_generator.generate_algorithm_appendix(algorithms)
    results_appendix = supp_generator.generate_extended_results({'convergence': True})
    
    supp_path = supp_generator.create_supplementary_document([algo_appendix, results_appendix])
    print(f"Generated supplementary material: {supp_path}")
    
    print("\nManuscript generation complete!")
    print("Files ready for journal submission.")


if __name__ == "__main__":
    demo_manuscript_generation()