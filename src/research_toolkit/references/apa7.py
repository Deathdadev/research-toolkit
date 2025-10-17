"""
APA 7th Edition Reference Manager - Enhanced Version

Comprehensive toolkit for managing and formatting references according to
APA 7th edition guidelines. Designed for integration with AI models and
research workflows.

Supports:
- Journal articles
- Books & book chapters
- Websites & online sources
- Conference papers
- Reports & technical documents
- Datasets
- Software
- Government documents
- Dissertations & theses
- And more...

Features:
- Multiple citation support
- Advanced author formatting (up to 20 authors)
- Proper title case conversion
- DOI and URL handling
- BibTeX export
- Comprehensive validation
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re


class APA7ReferenceManager:
    """
    Manages bibliographic references in APA 7th edition format.
    
    Enhanced features:
    - Support for 10+ reference types
    - Multiple citation handling
    - Advanced author formatting
    - Automatic title case conversion
    - Comprehensive validation
    - Export to multiple formats
    """
    
    # Required fields for each reference type
    REQUIRED_FIELDS = {
        'journal': ['author', 'year', 'title', 'journal', 'volume'],
        'book': ['author', 'year', 'title', 'publisher'],
        'chapter': ['author', 'year', 'title', 'editor', 'book_title', 'publisher'],
        'website': ['author', 'year', 'title', 'url'],
        'report': ['author', 'year', 'title', 'institution'],
        'conference': ['author', 'year', 'title', 'conference_name', 'location'],
        'dataset': ['author', 'year', 'title', 'publisher'],
        'software': ['author', 'year', 'title', 'publisher'],
        'dissertation': ['author', 'year', 'title', 'institution'],
        'government': ['author', 'year', 'title', 'department']
    }
    
    def __init__(self):
        """Initialize reference manager."""
        self.references = {}
        self.citation_count = 0
    
    @staticmethod
    def parse_author_name(name: str) -> Dict[str, str]:
        """
        Parse an author name into components (last name, first name, initials).
        
        Supports multiple formats:
        - "Smith, J. M." (Last, Initials)
        - "Smith, John Michael" (Last, First Middle)
        - "John Michael Smith" (First Middle Last)
        - "Smith" (Last only)
        - "J. M. Smith" (Initials Last)
        
        Args:
            name: Author name in various formats
            
        Returns:
            Dictionary with 'last', 'first', 'initials', 'formatted_citation', 'formatted_reference'
            
        Example:
            >>> parse_author_name("Smith, John Michael")
            {'last': 'Smith', 'first': 'John', 'middle': 'Michael', 
             'initials': 'J. M.', 'formatted_citation': 'Smith',
             'formatted_reference': 'Smith, J. M.'}
        """
        name = name.strip()
        result = {
            'last': '',
            'first': '',
            'middle': '',
            'initials': '',
            'formatted_citation': '',
            'formatted_reference': ''
        }
        
        # Handle empty
        if not name:
            return result
        
        # Format 1: "Last, First Middle" or "Last, J. M."
        if ',' in name:
            parts = name.split(',', 1)
            result['last'] = parts[0].strip()
            
            # Parse first/middle or initials
            remaining = parts[1].strip()
            
            # Check if already initials (contains periods)
            if '.' in remaining:
                result['initials'] = remaining
                # Try to extract first letter as first name
                initials_parts = remaining.replace('.', '').split()
                if initials_parts:
                    result['first'] = initials_parts[0]
                    if len(initials_parts) > 1:
                        result['middle'] = initials_parts[1]
            else:
                # Full names
                names = remaining.split()
                if names:
                    result['first'] = names[0]
                    if len(names) > 1:
                        result['middle'] = ' '.join(names[1:])
                    
                    # Generate initials
                    initials = [n[0] + '.' for n in names if n]
                    result['initials'] = ' '.join(initials)
        
        # Format 2: "First Middle Last" (no comma)
        else:
            parts = name.split()
            
            if len(parts) == 1:
                # Just last name
                result['last'] = parts[0]
            elif len(parts) == 2:
                # Could be "First Last" or "J. Last"
                if '.' in parts[0]:
                    # Initials Last
                    result['initials'] = parts[0]
                    result['last'] = parts[1]
                    result['first'] = parts[0].replace('.', '').strip()
                else:
                    # First Last
                    result['first'] = parts[0]
                    result['last'] = parts[1]
                    result['initials'] = parts[0][0] + '.'
            else:
                # 3+ parts: assume First Middle... Last
                result['first'] = parts[0]
                result['last'] = parts[-1]
                result['middle'] = ' '.join(parts[1:-1])
                
                # Generate initials from first and middle
                initials = [parts[0][0] + '.']
                for middle_part in parts[1:-1]:
                    if middle_part:
                        initials.append(middle_part[0] + '.')
                result['initials'] = ' '.join(initials)
        
        # Generate formatted versions
        result['formatted_citation'] = result['last'] if result['last'] else name
        
        if result['last'] and result['initials']:
            result['formatted_reference'] = f"{result['last']}, {result['initials']}"
        elif result['last'] and result['first']:
            # Generate initials from first and middle
            initials = [result['first'][0] + '.']
            if result['middle']:
                for part in result['middle'].split():
                    if part:
                        initials.append(part[0] + '.')
            result['initials'] = ' '.join(initials)
            result['formatted_reference'] = f"{result['last']}, {result['initials']}"
        else:
            result['formatted_reference'] = result['last'] if result['last'] else name
        
        return result
    
    @staticmethod
    def parse_multiple_authors(author_string: str) -> List[Dict[str, str]]:
        """
        Parse multiple authors from a string.
        
        Supports separators: semicolon (;), ampersand (&), "and"
        
        Args:
            author_string: String with multiple authors
            
        Returns:
            List of parsed author dictionaries
            
        Example:
            >>> parse_multiple_authors("Smith, J.; Jones, K. & Brown, M.")
            [{'last': 'Smith', 'initials': 'J.', ...},
             {'last': 'Jones', 'initials': 'K.', ...},
             {'last': 'Brown', 'initials': 'M.', ...}]
        """
        # Split by semicolon, ampersand, or " and "
        authors = re.split(r';\s*|&\s*|\sand\s', author_string)
        return [APA7ReferenceManager.parse_author_name(author.strip()) 
                for author in authors if author.strip()]
    
    @staticmethod
    def format_authors_for_citation(authors: List[Dict[str, str]]) -> str:
        """
        Format parsed authors for in-text citation.
        
        Rules:
        - 1 author: "Smith"
        - 2 authors: "Smith & Jones"
        - 3+ authors: "Smith et al."
        
        Args:
            authors: List of parsed author dictionaries
            
        Returns:
            Formatted citation string
        """
        if not authors:
            return 'N.p.'
        
        if len(authors) == 1:
            return authors[0]['formatted_citation']
        elif len(authors) == 2:
            return f"{authors[0]['formatted_citation']} & {authors[1]['formatted_citation']}"
        else:
            return f"{authors[0]['formatted_citation']} et al."
    
    @staticmethod
    def format_authors_for_reference(authors: List[Dict[str, str]]) -> str:
        """
        Format parsed authors for reference list.
        
        APA 7 rules:
        - Up to 20 authors: List all with & before last
        - 21+ authors: List first 19, ..., last
        
        Args:
            authors: List of parsed author dictionaries
            
        Returns:
            Formatted reference string
        """
        if not authors:
            return 'N.p.'
        
        formatted = [a['formatted_reference'] for a in authors]
        
        if len(formatted) <= 20:
            if len(formatted) == 1:
                return formatted[0]
            else:
                return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
        else:
            # 21+ authors: first 19, ellipsis, last
            return ", ".join(formatted[:19]) + f", ... {formatted[-1]}"
    
    def add_reference(
        self, 
        ref_type: str,
        citation_key: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Add a reference to the manager.
        
        Args:
            ref_type: Type of reference (journal, book, website, etc.)
            citation_key: Optional custom key (auto-generated if not provided)
            **kwargs: Reference details (author, year, title, etc.)
            
        Returns:
            Citation key for referencing
            
        Example:
            key = manager.add_reference(
                'journal',
                author='Smith, J. M.; Jones, K. L.',
                year='2023',
                title='Research Methods in the Digital Age',
                journal='Journal of Modern Research',
                volume='10',
                issue='2',
                pages='123-145',
                doi='10.1234/jmr.2023.123'
            )
        """
        if ref_type not in self.REQUIRED_FIELDS:
            raise ValueError(f"Unknown reference type: {ref_type}. "
                           f"Supported types: {', '.join(self.REQUIRED_FIELDS.keys())}")
        
        # Validate required fields
        required = self.REQUIRED_FIELDS[ref_type]
        missing = [f for f in required if f not in kwargs or not kwargs[f]]
        if missing:
            raise ValueError(
                f"Missing required fields for {ref_type}: {', '.join(missing)}\n"
                f"Required fields: {', '.join(required)}"
            )
        
        # Generate citation key if not provided
        if citation_key is None:
            self.citation_count += 1
            citation_key = f"ref{self.citation_count}"
        
        # Store reference
        self.references[citation_key] = {
            'type': ref_type,
            'details': kwargs
        }
        
        return citation_key
    
    def get_in_text_citation(
        self, 
        citation_keys: List[str],
        page: Optional[str] = None,
        narrative: bool = False
    ) -> str:
        """
        Generate in-text citation for one or more references.
        
        Args:
            citation_keys: List of reference keys (can be single key in list)
            page: Optional page number(s)
            narrative: If True, format as narrative citation
            
        Returns:
            Formatted in-text citation
            
        Examples:
            Parenthetical single: "(Smith, 2023)"
            Narrative single: "Smith (2023)"
            Parenthetical multiple: "(Smith, 2023; Jones, 2024)"
            With page: "(Smith, 2023, p. 45)"
        """
        if not isinstance(citation_keys, list):
            citation_keys = [citation_keys]
        
        citations = []
        for key in citation_keys:
            if key not in self.references:
                citations.append(f"[UNKNOWN: {key}]")
                continue
            
            ref = self.references[key]
            details = ref['details']
            
            author = self._format_author_citation(details.get('author', 'N.p.'))
            year = details.get('year', 'n.d.')
            
            citations.append(f"{author}, {year}")
        
        # Handle narrative vs parenthetical
        if narrative and len(citations) == 1:
            # Narrative format: "Author (year)"
            parts = citations[0].split(', ')
            citation_str = f"{parts[0]} ({parts[1]})"
        else:
            # Parenthetical format: "(Author, year; Author2, year2)"
            citation_str = f"({'; '.join(citations)})"
        
        # Add page numbers if provided
        if page:
            page_str = f"p. {page}" if '-' not in str(page) else f"pp. {page}"
            citation_str = citation_str.replace(")", f", {page_str})")
        
        return citation_str
    
    def _format_author_citation(self, author: str) -> str:
        """
        Format author name(s) for in-text citation.
        
        Handles:
        - Single author: "Smith"
        - Two authors: "Smith & Jones"
        - Three+ authors: "Smith et al."
        
        Uses enhanced name parsing for better handling of various formats.
        """
        if not author or author == 'N.p.':
            return 'N.p.'
        
        # Use new parsing methods
        authors = self.parse_multiple_authors(author)
        return self.format_authors_for_citation(authors)
    
    def _format_author_reference(self, author: str) -> str:
        """
        Format author name(s) for reference list.
        
        APA 7 format:
        - Up to 20 authors: List all with & before last
        - 21+ authors: List first 19, then ..., then last
        
        Uses enhanced name parsing for better handling of various formats.
        """
        if not author or author == 'N.p.':
            return 'N.p.'
        
        # Use new parsing methods
        authors = self.parse_multiple_authors(author)
        return self.format_authors_for_reference(authors)
    
    def _format_title(self, title: str, title_type: str = 'sentence') -> str:
        """
        Format title in sentence case or title case.
        
        Args:
            title: The title to format
            title_type: 'sentence' for sentence case, 'title' for title case
            
        Returns:
            Properly formatted title
        """
        if not title:
            return "[No title]"
        
        if title_type == 'sentence':
            # Sentence case: Only first word and proper nouns capitalized
            # Preserve capitalization after colons and periods
            sentences = re.split('([.:?!]\\s+)', title)
            formatted = []
            
            for i, part in enumerate(sentences):
                if i % 2 == 0 and part:  # Text parts
                    words = part.split()
                    if words:
                        # Capitalize first word
                        words[0] = words[0].capitalize()
                        formatted.append(' '.join(words))
                else:  # Punctuation
                    formatted.append(part)
            
            return ''.join(formatted)
        
        elif title_type == 'title':
            # Title case for journal names, book titles in references
            minor_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 
                          'nor', 'on', 'at', 'to', 'from', 'by', 'with', 'of', 'in'}
            words = title.split()
            formatted_words = []
            
            for i, word in enumerate(words):
                if i == 0 or word.lower() not in minor_words:
                    formatted_words.append(word.capitalize())
                else:
                    formatted_words.append(word.lower())
            
            return ' '.join(formatted_words)
        
        return title
    
    def _format_doi(self, doi: str) -> str:
        """Format DOI as a full URL."""
        if not doi:
            return ""
        if doi.startswith('http'):
            return doi
        return f"https://doi.org/{doi}"
    
    def _format_journal_article(self, details: Dict) -> str:
        """Format journal article reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        journal = self._format_title(details.get('journal', ''), 'title')
        volume = details.get('volume', '')
        issue = details.get('issue', '')
        pages = details.get('pages', '')
        doi = details.get('doi', '')
        url = details.get('url', '')
        
        ref = f"{author} ({year}). {title}. *{journal}*, *{volume}*"
        if issue:
            ref += f"({issue})"
        if pages:
            ref += f", {pages}"
        ref += "."
        
        # Add DOI or URL (prefer DOI)
        if doi:
            ref += f" {self._format_doi(doi)}"
        elif url:
            ref += f" {url}"
        
        return ref
    
    def _format_book(self, details: Dict) -> str:
        """Format book reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        edition = details.get('edition', '')
        publisher = details.get('publisher', '')
        location = details.get('location', '')
        doi = details.get('doi', '')
        url = details.get('url', '')
        
        ref = f"{author} ({year}). *{title}*"
        
        if edition:
            # Handle both "3rd" and "3" formats
            edition_str = edition if any(c.isalpha() for c in str(edition)) else f"{edition}th"
            ref += f" ({edition_str} ed.)"
        
        ref += "."
        
        if location:
            ref += f" {location}:"
        
        ref += f" {publisher}."
        
        if doi:
            ref += f" {self._format_doi(doi)}"
        elif url:
            ref += f" {url}"
        
        return ref
    
    def _format_chapter(self, details: Dict) -> str:
        """Format book chapter reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        editor = self._format_author_reference(details.get('editor', ''))
        book_title = self._format_title(details.get('book_title', ''), 'sentence')
        pages = details.get('pages', '')
        publisher = details.get('publisher', '')
        doi = details.get('doi', '')
        
        # Handle multiple editors
        ed_suffix = "(Eds.)" if ';' in details.get('editor', '') else "(Ed.)"
        
        ref = f"{author} ({year}). {title}. In {editor} {ed_suffix}, *{book_title}*"
        
        if pages:
            ref += f" (pp. {pages})"
        
        ref += f". {publisher}."
        
        if doi:
            ref += f" {self._format_doi(doi)}"
        
        return ref
    
    def _format_website(self, details: Dict) -> str:
        """Format website reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        website = details.get('website', '')
        url = details.get('url', '')
        retrieved = details.get('retrieved', '')
        
        ref = f"{author} ({year}). *{title}*."
        
        if website:
            ref += f" {website}."
        
        if retrieved:
            ref += f" Retrieved {retrieved}, from {url}"
        else:
            ref += f" {url}"
        
        return ref
    
    def _format_report(self, details: Dict) -> str:
        """Format report reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        report_number = details.get('report_number', '')
        institution = details.get('institution', '')
        url = details.get('url', '')
        
        ref = f"{author} ({year}). *{title}*"
        
        if report_number:
            ref += f" (Report No. {report_number})"
        
        ref += f". {institution}."
        
        if url:
            ref += f" {url}"
        
        return ref
    
    def _format_conference(self, details: Dict) -> str:
        """Format conference paper reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        conference_name = details.get('conference_name', '')
        location = details.get('location', '')
        url = details.get('url', '')
        
        ref = f"{author} ({year}). {title} [{conference_name}, {location}]."
        
        if url:
            ref += f" {url}"
        
        return ref
    
    def _format_dataset(self, details: Dict) -> str:
        """Format dataset reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        version = details.get('version', '')
        publisher = details.get('publisher', '')
        url = details.get('url', '')
        doi = details.get('doi', '')
        
        ref = f"{author} ({year}). *{title}*"
        
        if version:
            ref += f" (Version {version})"
        
        ref += f" [Data set]. {publisher}."
        
        if doi:
            ref += f" {self._format_doi(doi)}"
        elif url:
            ref += f" {url}"
        
        return ref
    
    def _format_software(self, details: Dict) -> str:
        """Format software reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        version = details.get('version', '')
        publisher = details.get('publisher', '')
        url = details.get('url', '')
        
        ref = f"{author} ({year}). *{title}*"
        
        if version:
            ref += f" (Version {version})"
        
        ref += f" [Computer software]. {publisher}."
        
        if url:
            ref += f" {url}"
        
        return ref
    
    def _format_dissertation(self, details: Dict) -> str:
        """Format dissertation reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        institution = details.get('institution', '')
        dissertation_type = details.get('dissertation_type', 'Doctoral dissertation')
        url = details.get('url', '')
        
        ref = f"{author} ({year}). *{title}* [{dissertation_type}, {institution}]."
        
        if url:
            ref += f" {url}"
        
        return ref
    
    def _format_government(self, details: Dict) -> str:
        """Format government document reference."""
        author = self._format_author_reference(details.get('author', ''))
        year = details.get('year', 'n.d.')
        title = self._format_title(details.get('title', ''), 'sentence')
        department = details.get('department', '')
        report_number = details.get('report_number', '')
        url = details.get('url', '')
        
        ref = f"{author} ({year}). *{title}*"
        
        if report_number:
            ref += f" ({report_number})"
        
        ref += f". {department}."
        
        if url:
            ref += f" {url}"
        
        return ref
    
    def format_reference(self, citation_key: str) -> str:
        """
        Format a single reference.
        
        Args:
            citation_key: Key of the reference
            
        Returns:
            Formatted reference string
        """
        if citation_key not in self.references:
            return f"[UNKNOWN REFERENCE: {citation_key}]"
        
        ref = self.references[citation_key]
        ref_type = ref['type']
        details = ref['details']
        
        # Route to appropriate formatter
        formatters = {
            'journal': self._format_journal_article,
            'book': self._format_book,
            'chapter': self._format_chapter,
            'website': self._format_website,
            'report': self._format_report,
            'conference': self._format_conference,
            'dataset': self._format_dataset,
            'software': self._format_software,
            'dissertation': self._format_dissertation,
            'government': self._format_government
        }
        
        formatter = formatters.get(ref_type)
        if formatter:
            return formatter(details)
        else:
            return f"[UNSUPPORTED TYPE: {ref_type}]"
    
    def generate_reference_list(self, sort: bool = True) -> str:
        """
        Generate complete reference list.
        
        Args:
            sort: If True, sort alphabetically by author
            
        Returns:
            Formatted reference list
        """
        if not self.references:
            return "No references added."
        
        formatted_refs = [self.format_reference(key) for key in self.references]
        
        if sort:
            formatted_refs.sort()
        
        return "\n\n".join(formatted_refs)
    
    def validate_reference(self, citation_key: str) -> Tuple[bool, List[str]]:
        """
        Validate a reference for completeness.
        
        Args:
            citation_key: Key of the reference
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if citation_key not in self.references:
            return False, ["Reference not found"]
        
        ref = self.references[citation_key]
        ref_type = ref['type']
        details = ref['details']
        
        issues = []
        
        # Check required fields
        required = self.REQUIRED_FIELDS.get(ref_type, [])
        for field in required:
            if field not in details or not details[field]:
                issues.append(f"Missing required field: {field}")
        
        # Check DOI format if present
        if 'doi' in details and details['doi']:
            doi = details['doi']
            if not (doi.startswith('10.') or doi.startswith('http')):
                issues.append("Invalid DOI format (should start with '10.' or be a full URL)")
        
        # Check year format
        if 'year' in details:
            year = str(details['year'])
            if not (year.isdigit() or year == 'n.d.'):
                issues.append("Invalid year format (should be 4 digits or 'n.d.')")
        
        # Check author format
        if 'author' in details and details['author']:
            author = details['author']
            if not re.search(r'[A-Za-z]+,\s*[A-Z]\.', author):
                issues.append("Author format may be incorrect (expected 'Last, F. M.')")
        
        return len(issues) == 0, issues
    
    def export_to_bibtex(self, citation_key: str) -> str:
        """
        Export reference to BibTeX format.
        
        Args:
            citation_key: Key of the reference
            
        Returns:
            BibTeX formatted entry
        """
        if citation_key not in self.references:
            return ""
        
        ref = self.references[citation_key]
        ref_type = ref['type']
        details = ref['details']
        
        # Map APA types to BibTeX types
        type_map = {
            'journal': 'article',
            'book': 'book',
            'chapter': 'incollection',
            'website': 'misc',
            'report': 'techreport',
            'conference': 'inproceedings',
            'dataset': 'misc',
            'software': 'misc',
            'dissertation': 'phdthesis',
            'government': 'techreport'
        }
        
        bib_type = type_map.get(ref_type, 'misc')
        
        # Build BibTeX entry
        bib_entry = f"@{bib_type}{{{citation_key},\n"
        
        # Convert author format for BibTeX
        if 'author' in details:
            authors = details['author'].split(';')
            bibtex_authors = " and ".join([a.strip() for a in authors])
            bib_entry += f"  author = {{{bibtex_authors}}},\n"
        
        # Add other fields
        for field, value in details.items():
            if field != 'author' and value:
                bib_entry += f"  {field} = {{{value}}},\n"
        
        bib_entry += "}\n"
        
        return bib_entry
    
    def clear_all(self):
        """Clear all references."""
        self.references = {}
        self.citation_count = 0
    
    def get_reference_count(self) -> int:
        """Get total number of references."""
        return len(self.references)
    
    def list_keys(self) -> List[str]:
        """Get list of all citation keys."""
        return list(self.references.keys())


__all__ = ['APA7ReferenceManager']
