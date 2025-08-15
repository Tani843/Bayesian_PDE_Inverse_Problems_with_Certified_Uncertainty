"""
Journal Submission and Review Management System

Automated tools for managing academic journal submissions, tracking review
processes, handling revisions, and maintaining submission records for
Bayesian PDE inverse problems research.

Features:
- Journal database with submission requirements
- Automated manuscript formatting for specific journals
- Submission tracking and timeline management
- Review process management and response generation
- Revision tracking and version control
- Impact factor and journal ranking analysis
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import logging


@dataclass
class JournalInfo:
    """Information about academic journal."""
    name: str
    abbreviation: str
    publisher: str
    impact_factor: float
    scope: List[str]
    submission_system: str
    typical_review_time: int  # weeks
    acceptance_rate: float
    open_access: bool
    submission_fee: float
    page_charges: float
    word_limit: int
    reference_limit: int
    figure_limit: int
    latex_template: str = ""
    special_requirements: List[str] = None
    
    def __post_init__(self):
        if self.special_requirements is None:
            self.special_requirements = []


@dataclass
class SubmissionRecord:
    """Record of journal submission."""
    submission_id: str
    journal: str
    manuscript_title: str
    authors: List[str]
    submission_date: datetime
    status: str  # submitted, under_review, revision_requested, accepted, rejected
    manuscript_version: str
    review_timeline: List[Dict]
    reviewer_comments: List[Dict]
    editor_decision: str = ""
    revision_deadline: Optional[datetime] = None
    final_decision_date: Optional[datetime] = None
    doi: str = ""
    publication_date: Optional[datetime] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'submission_id': self.submission_id,
            'journal': self.journal,
            'manuscript_title': self.manuscript_title,
            'authors': self.authors,
            'submission_date': self.submission_date.isoformat(),
            'status': self.status,
            'manuscript_version': self.manuscript_version,
            'review_timeline': self.review_timeline,
            'reviewer_comments': self.reviewer_comments,
            'editor_decision': self.editor_decision,
            'revision_deadline': self.revision_deadline.isoformat() if self.revision_deadline else None,
            'final_decision_date': self.final_decision_date.isoformat() if self.final_decision_date else None,
            'doi': self.doi,
            'publication_date': self.publication_date.isoformat() if self.publication_date else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary."""
        # Convert datetime strings back to datetime objects
        submission_date = datetime.fromisoformat(data['submission_date'])
        revision_deadline = datetime.fromisoformat(data['revision_deadline']) if data['revision_deadline'] else None
        final_decision_date = datetime.fromisoformat(data['final_decision_date']) if data['final_decision_date'] else None
        publication_date = datetime.fromisoformat(data['publication_date']) if data['publication_date'] else None
        
        return cls(
            submission_id=data['submission_id'],
            journal=data['journal'],
            manuscript_title=data['manuscript_title'],
            authors=data['authors'],
            submission_date=submission_date,
            status=data['status'],
            manuscript_version=data['manuscript_version'],
            review_timeline=data['review_timeline'],
            reviewer_comments=data['reviewer_comments'],
            editor_decision=data['editor_decision'],
            revision_deadline=revision_deadline,
            final_decision_date=final_decision_date,
            doi=data['doi'],
            publication_date=publication_date
        )


class JournalDatabase:
    """Database of academic journals with submission information."""
    
    def __init__(self):
        """Initialize journal database with computational mathematics journals."""
        self.journals = {
            'jcp': JournalInfo(
                name='Journal of Computational Physics',
                abbreviation='J. Comput. Phys.',
                publisher='Elsevier',
                impact_factor=4.645,
                scope=['computational physics', 'numerical methods', 'scientific computing'],
                submission_system='Editorial Manager',
                typical_review_time=12,
                acceptance_rate=0.25,
                open_access=False,
                submission_fee=0,
                page_charges=0,
                word_limit=15000,
                reference_limit=100,
                figure_limit=20,
                latex_template='elsarticle',
                special_requirements=['novelty statement', 'software availability']
            ),
            'siamsc': JournalInfo(
                name='SIAM Journal on Scientific Computing',
                abbreviation='SIAM J. Sci. Comput.',
                publisher='SIAM',
                impact_factor=3.817,
                scope=['scientific computing', 'numerical analysis', 'computational methods'],
                submission_system='SIAM Submission System',
                typical_review_time=16,
                acceptance_rate=0.30,
                open_access=False,
                submission_fee=0,
                page_charges=0,
                word_limit=12000,
                reference_limit=80,
                figure_limit=15,
                latex_template='siamltex',
                special_requirements=['reproducibility statement']
            ),
            'siamnum': JournalInfo(
                name='SIAM Journal on Numerical Analysis',
                abbreviation='SIAM J. Numer. Anal.',
                publisher='SIAM',
                impact_factor=2.912,
                scope=['numerical analysis', 'mathematical theory', 'convergence analysis'],
                submission_system='SIAM Submission System',
                typical_review_time=20,
                acceptance_rate=0.22,
                open_access=False,
                submission_fee=0,
                page_charges=0,
                word_limit=10000,
                reference_limit=60,
                figure_limit=10,
                latex_template='siamltex',
                special_requirements=['rigorous proofs', 'convergence analysis']
            ),
            'ip': JournalInfo(
                name='Inverse Problems',
                abbreviation='Inverse Problems',
                publisher='IOP Publishing',
                impact_factor=2.408,
                scope=['inverse problems', 'parameter estimation', 'ill-posed problems'],
                submission_system='ScholarOne Manuscripts',
                typical_review_time=14,
                acceptance_rate=0.28,
                open_access=True,
                submission_fee=0,
                page_charges=3500,  # USD for open access
                word_limit=12000,
                reference_limit=100,
                figure_limit=15,
                latex_template='iopart',
                special_requirements=['data availability statement']
            ),
            'juq': JournalInfo(
                name='SIAM/ASA Journal on Uncertainty Quantification',
                abbreviation='SIAM/ASA J. Uncertain. Quantif.',
                publisher='SIAM',
                impact_factor=2.031,
                scope=['uncertainty quantification', 'stochastic methods', 'risk analysis'],
                submission_system='SIAM Submission System',
                typical_review_time=18,
                acceptance_rate=0.35,
                open_access=False,
                submission_fee=0,
                page_charges=0,
                word_limit=10000,
                reference_limit=75,
                figure_limit=12,
                latex_template='siamltex',
                special_requirements=['uncertainty analysis']
            )
        }
    
    def get_journal(self, key: str) -> Optional[JournalInfo]:
        """Get journal information by key."""
        return self.journals.get(key)
    
    def search_journals(self, scope_keywords: List[str], 
                       min_impact_factor: float = 0) -> List[Tuple[str, JournalInfo]]:
        """
        Search journals by scope and impact factor.
        
        Parameters:
        -----------
        scope_keywords : List[str]
            Keywords to match against journal scope
        min_impact_factor : float
            Minimum impact factor threshold
            
        Returns:
        --------
        List[Tuple[str, JournalInfo]]
            Matching journals with keys
        """
        matches = []
        
        for key, journal in self.journals.items():
            # Check scope match
            scope_match = any(
                any(keyword.lower() in scope_item.lower() for scope_item in journal.scope)
                for keyword in scope_keywords
            )
            
            # Check impact factor
            if scope_match and journal.impact_factor >= min_impact_factor:
                matches.append((key, journal))
        
        # Sort by impact factor descending
        matches.sort(key=lambda x: x[1].impact_factor, reverse=True)
        return matches
    
    def get_submission_recommendations(self, manuscript_keywords: List[str],
                                     target_impact: float = 2.0) -> List[Dict]:
        """
        Get journal submission recommendations.
        
        Parameters:
        -----------
        manuscript_keywords : List[str]
            Keywords describing the manuscript
        target_impact : float
            Target impact factor
            
        Returns:
        --------
        List[Dict]
            Recommended journals with fit scores
        """
        recommendations = []
        
        for key, journal in self.journals.items():
            # Calculate scope fit score
            scope_matches = sum(
                any(keyword.lower() in scope_item.lower() for scope_item in journal.scope)
                for keyword in manuscript_keywords
            )
            fit_score = scope_matches / len(manuscript_keywords)
            
            # Impact factor alignment
            impact_alignment = 1.0 - abs(journal.impact_factor - target_impact) / max(journal.impact_factor, target_impact)
            
            # Combined score
            overall_score = 0.7 * fit_score + 0.3 * impact_alignment
            
            recommendations.append({
                'journal_key': key,
                'journal_name': journal.name,
                'impact_factor': journal.impact_factor,
                'fit_score': fit_score,
                'overall_score': overall_score,
                'review_time': journal.typical_review_time,
                'acceptance_rate': journal.acceptance_rate
            })
        
        # Sort by overall score
        recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
        return recommendations


class SubmissionManager:
    """
    Comprehensive submission and review management system.
    
    Tracks manuscript submissions, review processes, and publication status
    across multiple journals for research portfolio management.
    """
    
    def __init__(self, database_path: str = "submissions.json"):
        """
        Initialize submission manager.
        
        Parameters:
        -----------
        database_path : str
            Path to submission database file
        """
        self.database_path = Path(database_path)
        self.submissions: Dict[str, SubmissionRecord] = {}
        self.journal_db = JournalDatabase()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load existing submissions
        self.load_submissions()
    
    def load_submissions(self):
        """Load submission records from database."""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'r') as f:
                    data = json.load(f)
                
                for sub_id, sub_data in data.items():
                    submission = SubmissionRecord.from_dict(sub_data)
                    self.submissions[sub_id] = submission
                
                self.logger.info(f"Loaded {len(self.submissions)} submission records")
                
            except Exception as e:
                self.logger.error(f"Error loading submissions: {e}")
    
    def save_submissions(self):
        """Save submission records to database."""
        try:
            data = {}
            for sub_id, submission in self.submissions.items():
                data[sub_id] = submission.to_dict()
            
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.submissions)} submission records")
            
        except Exception as e:
            self.logger.error(f"Error saving submissions: {e}")
    
    def create_submission(self, journal_key: str, manuscript_title: str,
                         authors: List[str], manuscript_version: str = "v1") -> str:
        """
        Create new submission record.
        
        Parameters:
        -----------
        journal_key : str
            Journal identifier
        manuscript_title : str
            Title of manuscript
        authors : List[str]
            Author names
        manuscript_version : str
            Version identifier
            
        Returns:
        --------
        str
            Submission ID
        """
        # Generate submission ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_id = f"{journal_key}_{timestamp}"
        
        # Create submission record
        submission = SubmissionRecord(
            submission_id=submission_id,
            journal=journal_key,
            manuscript_title=manuscript_title,
            authors=authors,
            submission_date=datetime.now(),
            status="submitted",
            manuscript_version=manuscript_version,
            review_timeline=[{
                'date': datetime.now().isoformat(),
                'event': 'manuscript_submitted',
                'description': 'Initial submission to journal'
            }],
            reviewer_comments=[]
        )
        
        self.submissions[submission_id] = submission
        self.save_submissions()
        
        self.logger.info(f"Created submission {submission_id} to {journal_key}")
        return submission_id
    
    def update_submission_status(self, submission_id: str, new_status: str,
                               notes: str = "", update_timeline: bool = True):
        """
        Update submission status.
        
        Parameters:
        -----------
        submission_id : str
            Submission identifier
        new_status : str
            New status
        notes : str
            Additional notes
        update_timeline : bool
            Whether to update timeline
        """
        if submission_id not in self.submissions:
            self.logger.error(f"Submission {submission_id} not found")
            return
        
        submission = self.submissions[submission_id]
        old_status = submission.status
        submission.status = new_status
        
        if update_timeline:
            submission.review_timeline.append({
                'date': datetime.now().isoformat(),
                'event': f'status_change_{new_status}',
                'description': f'Status changed from {old_status} to {new_status}',
                'notes': notes
            })
        
        self.save_submissions()
        self.logger.info(f"Updated submission {submission_id} status: {old_status} -> {new_status}")
    
    def add_reviewer_feedback(self, submission_id: str, reviewer_id: str,
                            comments: str, recommendation: str, score: int = None):
        """
        Add reviewer feedback to submission.
        
        Parameters:
        -----------
        submission_id : str
            Submission identifier
        reviewer_id : str
            Reviewer identifier
        comments : str
            Reviewer comments
        recommendation : str
            Reviewer recommendation
        score : int, optional
            Numerical score if provided
        """
        if submission_id not in self.submissions:
            self.logger.error(f"Submission {submission_id} not found")
            return
        
        submission = self.submissions[submission_id]
        
        feedback = {
            'reviewer_id': reviewer_id,
            'date': datetime.now().isoformat(),
            'comments': comments,
            'recommendation': recommendation,
            'score': score
        }
        
        submission.reviewer_comments.append(feedback)
        
        # Update timeline
        submission.review_timeline.append({
            'date': datetime.now().isoformat(),
            'event': 'reviewer_feedback_received',
            'description': f'Feedback received from {reviewer_id}',
            'recommendation': recommendation
        })
        
        self.save_submissions()
        self.logger.info(f"Added reviewer feedback for submission {submission_id}")
    
    def generate_revision_response(self, submission_id: str) -> str:
        """
        Generate template for revision response letter.
        
        Parameters:
        -----------
        submission_id : str
            Submission identifier
            
        Returns:
        --------
        str
            Revision response template
        """
        if submission_id not in self.submissions:
            return "Submission not found"
        
        submission = self.submissions[submission_id]
        journal = self.journal_db.get_journal(submission.journal)
        
        response_template = f"""
Response to Reviewers

Manuscript: {submission.manuscript_title}
Journal: {journal.name if journal else submission.journal}
Submission ID: {submission.submission_id}

Dear Editor,

We thank you and the reviewers for the thorough and constructive review of our manuscript. We have carefully addressed all comments and suggestions, which have significantly improved the quality of our work.

Below, we provide a detailed point-by-point response to each reviewer's comments. Changes in the revised manuscript are highlighted in blue text.

"""
        
        # Add sections for each reviewer
        for i, feedback in enumerate(submission.reviewer_comments, 1):
            response_template += f"""
REVIEWER {i} COMMENTS AND RESPONSES:

Recommendation: {feedback['recommendation']}

"""
            
            # Split comments into individual points (simple heuristic)
            comments = feedback['comments']
            comment_points = [c.strip() for c in comments.split('\n') if c.strip()]
            
            for j, comment in enumerate(comment_points[:5], 1):  # Limit to first 5 points
                response_template += f"""
Comment {i}.{j}: {comment[:200]}{'...' if len(comment) > 200 else ''}

Response: [Please provide detailed response addressing this comment]

Changes made: [Describe specific changes made to the manuscript]

"""
        
        response_template += """
We believe that these revisions have substantially improved the manuscript and hope that it is now suitable for publication in your journal.

Thank you for your consideration.

Sincerely,
[Author names]
"""
        
        return response_template
    
    def track_submission_timeline(self, submission_id: str) -> Dict[str, Any]:
        """
        Analyze submission timeline and predict next steps.
        
        Parameters:
        -----------
        submission_id : str
            Submission identifier
            
        Returns:
        --------
        Dict[str, Any]
            Timeline analysis
        """
        if submission_id not in self.submissions:
            return {"error": "Submission not found"}
        
        submission = self.submissions[submission_id]
        journal = self.journal_db.get_journal(submission.journal)
        
        # Calculate time elapsed
        days_since_submission = (datetime.now() - submission.submission_date).days
        
        analysis = {
            'submission_id': submission_id,
            'days_since_submission': days_since_submission,
            'current_status': submission.status,
            'timeline_events': len(submission.review_timeline),
            'reviewer_feedback_count': len(submission.reviewer_comments)
        }
        
        if journal:
            expected_review_weeks = journal.typical_review_time
            expected_days = expected_review_weeks * 7
            
            analysis['expected_review_time_weeks'] = expected_review_weeks
            analysis['expected_completion_date'] = (
                submission.submission_date + timedelta(days=expected_days)
            ).isoformat()
            
            if days_since_submission > expected_days:
                analysis['status_note'] = "Review time exceeded expectation"
            else:
                remaining_days = expected_days - days_since_submission
                analysis['estimated_days_remaining'] = remaining_days
        
        return analysis
    
    def generate_submission_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive submission portfolio report.
        
        Returns:
        --------
        Dict[str, Any]
            Submission portfolio analysis
        """
        report = {
            'total_submissions': len(self.submissions),
            'status_distribution': {},
            'journal_distribution': {},
            'timeline_analysis': {},
            'success_metrics': {}
        }
        
        # Status distribution
        for submission in self.submissions.values():
            status = submission.status
            report['status_distribution'][status] = report['status_distribution'].get(status, 0) + 1
        
        # Journal distribution
        for submission in self.submissions.values():
            journal = submission.journal
            report['journal_distribution'][journal] = report['journal_distribution'].get(journal, 0) + 1
        
        # Timeline analysis
        review_times = []
        for submission in self.submissions.values():
            if submission.status in ['accepted', 'rejected']:
                if submission.final_decision_date:
                    days = (submission.final_decision_date - submission.submission_date).days
                    review_times.append(days)
        
        if review_times:
            report['timeline_analysis'] = {
                'average_review_time_days': np.mean(review_times),
                'median_review_time_days': np.median(review_times),
                'min_review_time_days': min(review_times),
                'max_review_time_days': max(review_times)
            }
        
        # Success metrics
        accepted = sum(1 for s in self.submissions.values() if s.status == 'accepted')
        rejected = sum(1 for s in self.submissions.values() if s.status == 'rejected')
        total_decided = accepted + rejected
        
        if total_decided > 0:
            report['success_metrics'] = {
                'acceptance_rate': accepted / total_decided,
                'total_accepted': accepted,
                'total_rejected': rejected,
                'papers_published': sum(1 for s in self.submissions.values() if s.publication_date)
            }
        
        return report
    
    def visualize_submission_portfolio(self, save_path: str = None):
        """
        Create visualization of submission portfolio.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save visualization
        """
        if not self.submissions:
            print("No submissions to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Status distribution
        status_counts = {}
        for submission in self.submissions.values():
            status = submission.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            axes[0, 0].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Submission Status Distribution')
        
        # Journal distribution
        journal_counts = {}
        for submission in self.submissions.values():
            journal = submission.journal
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
        
        if journal_counts:
            axes[0, 1].bar(journal_counts.keys(), journal_counts.values())
            axes[0, 1].set_title('Submissions by Journal')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Timeline analysis
        submission_dates = [s.submission_date for s in self.submissions.values()]
        if submission_dates:
            # Group by month
            monthly_counts = {}
            for date in submission_dates:
                month_key = date.strftime('%Y-%m')
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
            
            axes[1, 0].plot(list(monthly_counts.keys()), list(monthly_counts.values()), marker='o')
            axes[1, 0].set_title('Submissions Over Time')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Review time analysis
        review_times = []
        journals = []
        for submission in self.submissions.values():
            days_elapsed = (datetime.now() - submission.submission_date).days
            review_times.append(days_elapsed)
            journals.append(submission.journal)
        
        if review_times:
            unique_journals = list(set(journals))
            avg_times = []
            for journal in unique_journals:
                journal_times = [review_times[i] for i, j in enumerate(journals) if j == journal]
                avg_times.append(np.mean(journal_times))
            
            axes[1, 1].bar(unique_journals, avg_times)
            axes[1, 1].set_title('Average Review Time by Journal')
            axes[1, 1].set_ylabel('Days')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio visualization saved to {save_path}")
        else:
            plt.show()


def demo_submission_management():
    """Demonstrate submission management capabilities."""
    print("Journal Submission Management Demo")
    print("=" * 35)
    
    # Initialize manager
    manager = SubmissionManager("demo_submissions.json")
    
    # Get journal recommendations
    print("Getting journal recommendations...")
    keywords = ['bayesian', 'inverse problems', 'uncertainty quantification']
    recommendations = manager.journal_db.get_submission_recommendations(keywords, target_impact=3.0)
    
    print("Top journal recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec['journal_name']} (IF: {rec['impact_factor']}, Score: {rec['overall_score']:.2f})")
    
    # Create sample submissions
    print("\nCreating sample submissions...")
    
    submissions = [
        {
            'journal': 'jcp',
            'title': 'Certified Uncertainty Quantification for Bayesian PDE Inverse Problems',
            'authors': ['Jane Researcher', 'John Collaborator']
        },
        {
            'journal': 'siamsc',
            'title': 'Adaptive MCMC Methods for Large-Scale PDE Parameter Estimation',
            'authors': ['Jane Researcher', 'Maria Theorist']
        }
    ]
    
    submission_ids = []
    for sub in submissions:
        sub_id = manager.create_submission(
            journal_key=sub['journal'],
            manuscript_title=sub['title'],
            authors=sub['authors']
        )
        submission_ids.append(sub_id)
        print(f"Created submission: {sub_id}")
    
    # Simulate review process
    print("\nSimulating review process...")
    for sub_id in submission_ids:
        # Update to under review
        manager.update_submission_status(sub_id, 'under_review', 'Manuscript sent to reviewers')
        
        # Add reviewer feedback
        manager.add_reviewer_feedback(
            sub_id,
            'Reviewer 1',
            'The paper presents interesting results but needs clarification on the convergence analysis.',
            'minor_revision',
            score=3
        )
        
        manager.add_reviewer_feedback(
            sub_id,
            'Reviewer 2',
            'Well-written paper with solid theoretical contributions. Some experimental validation would strengthen the work.',
            'accept',
            score=4
        )
        
        # Update to revision requested
        manager.update_submission_status(sub_id, 'revision_requested', 'Minor revisions required')
    
    # Generate timeline analysis
    print("\nTimeline analysis:")
    for sub_id in submission_ids:
        timeline = manager.track_submission_timeline(sub_id)
        print(f"Submission {sub_id}: {timeline['days_since_submission']} days since submission")
    
    # Generate revision response template
    print("\nGenerating revision response template...")
    response = manager.generate_revision_response(submission_ids[0])
    print("Response template generated (first 500 characters):")
    print(response[:500] + "...")
    
    # Generate portfolio report
    print("\nGenerating submission portfolio report...")
    report = manager.generate_submission_report()
    print(f"Total submissions: {report['total_submissions']}")
    print(f"Status distribution: {report['status_distribution']}")
    
    # Save data
    manager.save_submissions()
    
    print("\nSubmission management demo complete!")


if __name__ == "__main__":
    demo_submission_management()