"""
Citation Management and Bibliography Tools

Automated citation management system for academic research in Bayesian PDE
inverse problems. Provides tools for reference collection, citation formatting,
impact tracking, and collaboration management.

Features:
- Automated DOI-based reference retrieval
- BibTeX generation and management
- Citation network analysis
- Impact metrics tracking
- Collaboration mapping
- Journal-specific formatting
"""

import requests
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
import habanero
from scholarly import scholarly
import time


@dataclass
class Reference:
    """Data class for academic reference information."""
    title: str
    authors: List[str]
    year: int
    journal: str = ""
    volume: str = ""
    pages: str = ""
    doi: str = ""
    arxiv_id: str = ""
    citation_count: int = 0
    bibtex_key: str = ""
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if not self.bibtex_key:
            self.bibtex_key = self.generate_bibtex_key()
    
    def generate_bibtex_key(self) -> str:
        """Generate BibTeX citation key from reference data."""
        if not self.authors:
            first_author = "unknown"
        else:
            first_author = self.authors[0].split()[-1].lower()  # Last name
            first_author = re.sub(r'[^a-z]', '', first_author)
        
        # Clean title for key generation
        title_words = re.findall(r'\b\w+\b', self.title.lower())
        key_words = [w for w in title_words[:3] if len(w) > 3]
        title_part = ''.join(key_words[:2]) if key_words else "work"
        
        return f"{first_author}{self.year}{title_part}"


class CitationManager:
    """
    Comprehensive citation management system for academic research.
    
    Manages reference collection, citation formatting, impact tracking,
    and collaboration analysis for Bayesian PDE research publications.
    """
    
    def __init__(self, database_path: str = "citations.json"):
        """
        Initialize citation manager.
        
        Parameters:
        -----------
        database_path : str
            Path to citation database file
        """
        self.database_path = Path(database_path)
        self.references: Dict[str, Reference] = {}
        self.citation_network = nx.DiGraph()
        
        # Initialize Crossref client for DOI lookups
        self.crossref = habanero.Crossref()
        
        # Load existing database
        self.load_database()
        
        # Research categories for automatic classification
        self.research_categories = {
            'bayesian_inference': [
                'bayesian', 'posterior', 'prior', 'mcmc', 'gibbs', 'metropolis'
            ],
            'inverse_problems': [
                'inverse problem', 'parameter estimation', 'ill-posed', 'regularization'
            ],
            'uncertainty_quantification': [
                'uncertainty', 'confidence', 'credible', 'prediction interval'
            ],
            'pde_methods': [
                'partial differential', 'finite element', 'finite difference', 'spectral'
            ],
            'optimization': [
                'optimization', 'gradient', 'newton', 'quasi-newton', 'trust region'
            ],
            'machine_learning': [
                'neural network', 'deep learning', 'gaussian process', 'kernel'
            ]
        }
    
    def load_database(self):
        """Load citation database from file."""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for key, ref_data in data.items():
                    ref = Reference(**ref_data)
                    self.references[key] = ref
                
                print(f"Loaded {len(self.references)} references from database")
                
            except Exception as e:
                print(f"Error loading database: {e}")
    
    def save_database(self):
        """Save citation database to file."""
        try:
            data = {}
            for key, ref in self.references.items():
                data[key] = {
                    'title': ref.title,
                    'authors': ref.authors,
                    'year': ref.year,
                    'journal': ref.journal,
                    'volume': ref.volume,
                    'pages': ref.pages,
                    'doi': ref.doi,
                    'arxiv_id': ref.arxiv_id,
                    'citation_count': ref.citation_count,
                    'bibtex_key': ref.bibtex_key,
                    'categories': ref.categories
                }
            
            with open(self.database_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(self.references)} references to database")
            
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def add_reference_by_doi(self, doi: str) -> Optional[Reference]:
        """
        Add reference by DOI lookup.
        
        Parameters:
        -----------
        doi : str
            DOI identifier
            
        Returns:
        --------
        Optional[Reference]
            Added reference or None if failed
        """
        try:
            # Query Crossref for DOI information
            work = self.crossref.works(ids=doi)
            item = work['message']
            
            # Extract information
            title = item.get('title', ['Unknown Title'])[0]
            
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)
            
            # Extract publication year
            published = item.get('published-print') or item.get('published-online')
            year = published['date-parts'][0][0] if published else 0
            
            # Journal information
            journal = item.get('container-title', [''])[0]
            volume = item.get('volume', '')
            pages = item.get('page', '')
            
            # Create reference
            ref = Reference(
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                volume=volume,
                pages=pages,
                doi=doi
            )
            
            # Categorize reference
            ref.categories = self.categorize_reference(ref)
            
            # Add to database
            self.references[ref.bibtex_key] = ref
            
            print(f"Added reference: {ref.title}")
            return ref
            
        except Exception as e:
            print(f"Error retrieving DOI {doi}: {e}")
            return None
    
    def add_reference_manual(self, title: str, authors: List[str], year: int,
                           journal: str = "", **kwargs) -> Reference:
        """
        Manually add reference.
        
        Parameters:
        -----------
        title : str
            Paper title
        authors : List[str]
            Author names
        year : int
            Publication year
        journal : str
            Journal name
        **kwargs : Dict
            Additional reference fields
            
        Returns:
        --------
        Reference
            Added reference
        """
        ref = Reference(
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            **kwargs
        )
        
        # Categorize and add
        ref.categories = self.categorize_reference(ref)
        self.references[ref.bibtex_key] = ref
        
        print(f"Added reference: {title}")
        return ref
    
    def categorize_reference(self, ref: Reference) -> List[str]:
        """
        Automatically categorize reference based on title and abstract.
        
        Parameters:
        -----------
        ref : Reference
            Reference to categorize
            
        Returns:
        --------
        List[str]
            Assigned categories
        """
        text = (ref.title + " " + ref.journal).lower()
        categories = []
        
        for category, keywords in self.research_categories.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def search_references(self, query: str, max_results: int = 20) -> List[Reference]:
        """
        Search Google Scholar for references and add to database.
        
        Parameters:
        -----------
        query : str
            Search query
        max_results : int
            Maximum number of results
            
        Returns:
        --------
        List[Reference]
            Found references
        """
        try:
            search_query = scholarly.search_pubs(query)
            found_refs = []
            
            for i, pub in enumerate(search_query):
                if i >= max_results:
                    break
                
                try:
                    # Extract publication info
                    title = pub.get('title', 'Unknown Title')
                    authors = [author.split()[-1] for author in pub.get('author', [])]
                    year = int(pub.get('year', 0)) if pub.get('year') else 0
                    journal = pub.get('journal', '')
                    
                    # Create reference
                    ref = Reference(
                        title=title,
                        authors=authors,
                        year=year,
                        journal=journal,
                        citation_count=int(pub.get('num_citations', 0))
                    )
                    
                    ref.categories = self.categorize_reference(ref)
                    
                    # Check if already exists
                    if ref.bibtex_key not in self.references:
                        self.references[ref.bibtex_key] = ref
                        found_refs.append(ref)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing publication: {e}")
                    continue
            
            print(f"Found and added {len(found_refs)} new references")
            return found_refs
            
        except Exception as e:
            print(f"Error searching references: {e}")
            return []
    
    def generate_bibtex(self, keys: List[str] = None, 
                       categories: List[str] = None) -> str:
        """
        Generate BibTeX bibliography.
        
        Parameters:
        -----------
        keys : List[str], optional
            Specific reference keys to include
        categories : List[str], optional
            Categories to include
            
        Returns:
        --------
        str
            BibTeX formatted bibliography
        """
        if keys:
            refs_to_include = [self.references[key] for key in keys 
                             if key in self.references]
        elif categories:
            refs_to_include = [ref for ref in self.references.values()
                             if any(cat in ref.categories for cat in categories)]
        else:
            refs_to_include = list(self.references.values())
        
        bib_entries = []
        
        for ref in refs_to_include:
            # Determine entry type
            entry_type = "article" if ref.journal else "misc"
            
            # Format authors
            author_str = " and ".join(ref.authors)
            
            # Create BibTeX entry
            entry = f"@{entry_type}{{{ref.bibtex_key},\n"
            entry += f"  title = {{{ref.title}}},\n"
            entry += f"  author = {{{author_str}}},\n"
            entry += f"  year = {{{ref.year}}},\n"
            
            if ref.journal:
                entry += f"  journal = {{{ref.journal}}},\n"
            if ref.volume:
                entry += f"  volume = {{{ref.volume}}},\n"
            if ref.pages:
                entry += f"  pages = {{{ref.pages}}},\n"
            if ref.doi:
                entry += f"  doi = {{{ref.doi}}},\n"
            
            entry = entry.rstrip(',\n') + "\n}\n\n"
            bib_entries.append(entry)
        
        return ''.join(bib_entries)
    
    def export_bibtex_file(self, filename: str, **kwargs):
        """Export BibTeX to file."""
        bibtex_content = self.generate_bibtex(**kwargs)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(bibtex_content)
        
        print(f"Exported BibTeX to {filename}")
    
    def analyze_citation_network(self) -> Dict[str, Any]:
        """
        Analyze citation patterns and collaboration networks.
        
        Returns:
        --------
        Dict[str, Any]
            Network analysis results
        """
        # Build citation network
        G = nx.Graph()
        
        # Add nodes for authors
        all_authors = set()
        for ref in self.references.values():
            all_authors.update(ref.authors)
        
        G.add_nodes_from(all_authors)
        
        # Add edges for co-authorship
        for ref in self.references.values():
            authors = ref.authors
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    if G.has_edge(authors[i], authors[j]):
                        G[authors[i]][authors[j]]['weight'] += 1
                    else:
                        G.add_edge(authors[i], authors[j], weight=1)
        
        # Network analysis
        analysis = {
            'num_authors': G.number_of_nodes(),
            'num_collaborations': G.number_of_edges(),
            'density': nx.density(G),
            'clustering': nx.average_clustering(G),
            'components': nx.number_connected_components(G)
        }
        
        # Top collaborators
        if G.number_of_edges() > 0:
            degree_centrality = nx.degree_centrality(G)
            top_collaborators = sorted(degree_centrality.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
            analysis['top_collaborators'] = top_collaborators
        
        return analysis
    
    def generate_citation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive citation and impact report.
        
        Returns:
        --------
        Dict[str, Any]
            Citation analysis report
        """
        report = {
            'total_references': len(self.references),
            'categories': {},
            'year_distribution': {},
            'journal_distribution': {},
            'top_cited': [],
            'recent_additions': []
        }
        
        # Category analysis
        all_categories = set()
        for ref in self.references.values():
            all_categories.update(ref.categories)
        
        for category in all_categories:
            count = sum(1 for ref in self.references.values() 
                       if category in ref.categories)
            report['categories'][category] = count
        
        # Year distribution
        for ref in self.references.values():
            year = ref.year
            report['year_distribution'][year] = report['year_distribution'].get(year, 0) + 1
        
        # Journal distribution
        for ref in self.references.values():
            if ref.journal:
                journal = ref.journal
                report['journal_distribution'][journal] = report['journal_distribution'].get(journal, 0) + 1
        
        # Top cited papers
        cited_refs = [(ref.citation_count, ref.title, ref.bibtex_key) 
                     for ref in self.references.values() if ref.citation_count > 0]
        report['top_cited'] = sorted(cited_refs, reverse=True)[:10]
        
        return report
    
    def visualize_citation_trends(self, save_path: str = None):
        """
        Create visualization of citation trends and patterns.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Year distribution
        years = []
        counts = []
        year_dist = {}
        for ref in self.references.values():
            year_dist[ref.year] = year_dist.get(ref.year, 0) + 1
        
        sorted_years = sorted(year_dist.items())
        years, counts = zip(*sorted_years) if sorted_years else ([], [])
        
        axes[0, 0].bar(years, counts, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Publications by Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Publications')
        
        # Category distribution
        categories = {}
        for ref in self.references.values():
            for cat in ref.categories:
                categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            cat_names, cat_counts = zip(*sorted(categories.items(), key=lambda x: x[1], reverse=True)[:8])
            axes[0, 1].pie(cat_counts, labels=cat_names, autopct='%1.1f%%')
            axes[0, 1].set_title('Research Categories')
        
        # Citation distribution
        citations = [ref.citation_count for ref in self.references.values() if ref.citation_count > 0]
        if citations:
            axes[1, 0].hist(citations, bins=20, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Citation Count Distribution')
            axes[1, 0].set_xlabel('Citations')
            axes[1, 0].set_ylabel('Number of Papers')
        
        # Top journals
        journals = {}
        for ref in self.references.values():
            if ref.journal:
                journals[ref.journal] = journals.get(ref.journal, 0) + 1
        
        if journals:
            top_journals = sorted(journals.items(), key=lambda x: x[1], reverse=True)[:10]
            journal_names, journal_counts = zip(*top_journals)
            
            axes[1, 1].barh(range(len(journal_names)), journal_counts, alpha=0.7, color='orange')
            axes[1, 1].set_yticks(range(len(journal_names)))
            axes[1, 1].set_yticklabels([j[:30] + '...' if len(j) > 30 else j for j in journal_names])
            axes[1, 1].set_title('Top Journals')
            axes[1, 1].set_xlabel('Number of Papers')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Citation trends visualization saved to {save_path}")
        else:
            plt.show()


def demo_citation_management():
    """Demonstrate citation management capabilities."""
    print("Citation Management Demo")
    print("=" * 30)
    
    # Initialize manager
    manager = CitationManager("demo_citations.json")
    
    # Add some sample references manually
    bayesian_refs = [
        {
            'title': 'Inverse problems: a Bayesian perspective',
            'authors': ['Andrew M. Stuart'],
            'year': 2010,
            'journal': 'Acta Numerica',
            'volume': '19',
            'pages': '451-559'
        },
        {
            'title': 'Statistical and computational inverse problems',
            'authors': ['Jari Kaipio', 'Erkki Somersalo'],
            'year': 2005,
            'journal': 'Applied Mathematical Sciences'
        },
        {
            'title': 'The Bayesian approach to inverse problems',
            'authors': ['Masoumeh Dashti', 'Andrew M. Stuart'],
            'year': 2017,
            'journal': 'Handbook of Uncertainty Quantification'
        }
    ]
    
    print("Adding sample references...")
    for ref_data in bayesian_refs:
        manager.add_reference_manual(**ref_data)
    
    # Search for additional references
    print("\nSearching for Bayesian PDE references...")
    found_refs = manager.search_references("Bayesian PDE inverse problems", max_results=5)
    
    # Generate citation report
    print("\nGenerating citation report...")
    report = manager.generate_citation_report()
    
    print(f"Total references: {report['total_references']}")
    print(f"Categories: {list(report['categories'].keys())}")
    print(f"Year range: {min(report['year_distribution'].keys()) if report['year_distribution'] else 'N/A'} - {max(report['year_distribution'].keys()) if report['year_distribution'] else 'N/A'}")
    
    # Export BibTeX
    print("\nExporting BibTeX...")
    manager.export_bibtex_file("bayesian_pde_refs.bib", categories=['bayesian_inference', 'inverse_problems'])
    
    # Generate network analysis
    print("\nAnalyzing collaboration network...")
    network_analysis = manager.analyze_citation_network()
    print(f"Authors: {network_analysis['num_authors']}")
    print(f"Collaborations: {network_analysis['num_collaborations']}")
    
    # Save database
    manager.save_database()
    
    print("\nCitation management demo complete!")


if __name__ == "__main__":
    demo_citation_management()