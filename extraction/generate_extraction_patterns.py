#!/usr/bin/env python3
"""
Generate extraction patterns for different markdown document types in data/misc folder.
Analyzes document structure and creates appropriate extraction patterns.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class DocumentPattern:
    """Represents an extraction pattern for a document type"""
    name: str
    description: str
    header_patterns: List[str]
    section_patterns: List[str]
    metadata_patterns: List[str]
    content_indicators: List[str]
    special_features: List[str]

class PatternGenerator:
    def __init__(self, data_dir: str = "data/misc"):
        self.data_dir = Path(data_dir)
        self.patterns = {}
        
    def analyze_document(self, file_path: Path) -> Dict:
        """Analyze a single markdown document and extract structural patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except:
                return {"error": "Could not read file"}
        
        analysis = {
            "filename": file_path.name,
            "headers": self._extract_headers(content),
            "sections": self._extract_sections(content),
            "metadata": self._extract_metadata(content),
            "special_elements": self._extract_special_elements(content),
            "document_type": self._classify_document_type(content, file_path.name),
            "language": self._detect_language(content),
            "structure_complexity": self._assess_complexity(content)
        }
        
        return analysis
    
    def _extract_headers(self, content: str) -> List[Dict]:
        """Extract all header patterns from content"""
        headers = []
        
        # Markdown headers
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
            headers.append({
                "type": "markdown",
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
                "pattern": f"{'#' * len(match.group(1))} {match.group(2).strip()}"
            })
        
        # Underlined headers
        lines = content.split('\n')
        for i, line in enumerate(lines[:-1]):
            next_line = lines[i + 1]
            if re.match(r'^=+$', next_line.strip()) and line.strip():
                headers.append({
                    "type": "underlined_h1",
                    "level": 1,
                    "text": line.strip(),
                    "pattern": f"{line.strip()}\n{'=' * len(next_line.strip())}"
                })
            elif re.match(r'^-+$', next_line.strip()) and line.strip():
                headers.append({
                    "type": "underlined_h2", 
                    "level": 2,
                    "text": line.strip(),
                    "pattern": f"{line.strip()}\n{'-' * len(next_line.strip())}"
                })
        
        return headers
    
    def _extract_sections(self, content: str) -> List[Dict]:
        """Extract section patterns"""
        sections = []
        
        # Table of contents
        if re.search(r'(table of contents|index|目录)', content, re.IGNORECASE):
            sections.append({"type": "table_of_contents", "indicator": "TOC present"})
        
        # Numbered sections
        numbered_sections = re.findall(r'^\s*(\d+\.?\d*\.?\d*)\s+(.+)$', content, re.MULTILINE)
        if numbered_sections:
            sections.append({
                "type": "numbered_sections",
                "count": len(numbered_sections),
                "examples": numbered_sections[:3]
            })
        
        # Lettered sections (A, B, C or a, b, c)
        lettered_sections = re.findall(r'^\s*([A-Za-z]\.)\s+(.+)$', content, re.MULTILINE)
        if lettered_sections:
            sections.append({
                "type": "lettered_sections", 
                "count": len(lettered_sections),
                "examples": lettered_sections[:3]
            })
        
        # Roman numerals
        roman_sections = re.findall(r'^\s*([IVX]+\.)\s+(.+)$', content, re.MULTILINE)
        if roman_sections:
            sections.append({
                "type": "roman_sections",
                "count": len(roman_sections),
                "examples": roman_sections[:3]
            })
        
        return sections
    
    def _extract_metadata(self, content: str) -> Dict:
        """Extract metadata patterns"""
        metadata = {}
        
        # Dates
        dates = re.findall(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', content)
        if dates:
            metadata["dates"] = dates[:5]  # First 5 dates
        
        # Phone numbers
        phones = re.findall(r'[\+]?[\d\s\-\(\)]{10,}', content)
        if phones:
            metadata["phones"] = [p.strip() for p in phones[:3]]
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        if emails:
            metadata["emails"] = emails[:3]
        
        # URLs
        urls = re.findall(r'https?://[^\s<>"]+', content)
        if urls:
            metadata["urls"] = urls[:3]
        
        # Reference numbers (various formats)
        ref_numbers = re.findall(r'\b[A-Z]{2,}\d{3,}|\b\d{4,}[A-Z]{2,}|\b[A-Z]\d{6,}', content)
        if ref_numbers:
            metadata["reference_numbers"] = ref_numbers[:5]
        
        return metadata
    
    def _extract_special_elements(self, content: str) -> List[str]:
        """Extract special document elements"""
        elements = []
        
        # Tables
        if '|' in content and re.search(r'\|.*\|.*\|', content):
            elements.append("tables")
        
        # Lists
        if re.search(r'^\s*[-*+]\s+', content, re.MULTILINE):
            elements.append("bullet_lists")
        
        if re.search(r'^\s*\d+\.\s+', content, re.MULTILINE):
            elements.append("numbered_lists")
        
        # Code blocks
        if '```' in content or '    ' in content:
            elements.append("code_blocks")
        
        # Images
        if re.search(r'!\[.*\]\(.*\)|<!-- image -->', content):
            elements.append("images")
        
        # Emphasis
        if re.search(r'\*\*.*\*\*|__.*__|_.*_|\*.*\*', content):
            elements.append("emphasis")
        
        # Links
        if re.search(r'\[.*\]\(.*\)', content):
            elements.append("links")
        
        # Blockquotes
        if re.search(r'^\s*>', content, re.MULTILINE):
            elements.append("blockquotes")
        
        # Horizontal rules
        if re.search(r'^---+$|^\*\*\*+$', content, re.MULTILINE):
            elements.append("horizontal_rules")
        
        return elements
    
    def _classify_document_type(self, content: str, filename: str) -> str:
        """Classify the document type based on content and filename"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Financial/Banking documents
        if any(word in content_lower for word in ['account', 'payment', 'transfer', 'bank', 'credit', 'debit']):
            return "financial"
        
        # Legal documents
        if any(word in content_lower for word in ['agreement', 'contract', 'terms', 'conditions', 'legal', 'clause']):
            return "legal"
        
        # Corporate/Policy documents
        if any(word in content_lower for word in ['policy', 'code of conduct', 'employee', 'privacy notice', 'compliance']):
            return "corporate_policy"
        
        # Travel/Tourism documents
        if any(word in content_lower for word in ['travel', 'hotel', 'flight', 'tourism', 'guide', 'city']):
            return "travel_tourism"
        
        # Technical/Manual documents
        if any(word in content_lower for word in ['how to', 'manual', 'guide', 'instructions', 'tutorial']):
            return "technical_manual"
        
        # Medical/Health documents
        if any(word in content_lower for word in ['health', 'medical', 'patient', 'treatment', 'diagnosis']):
            return "medical_health"
        
        # Government/Official documents
        if any(word in content_lower for word in ['government', 'official', 'department', 'ministry', 'registration']):
            return "government_official"
        
        # Academic/Research documents
        if any(word in content_lower for word in ['research', 'study', 'analysis', 'report', 'findings']):
            return "academic_research"
        
        # Form documents
        if 'form' in filename_lower or any(word in content_lower for word in ['application', 'form', 'declaration']):
            return "form"
        
        return "general"
    
    def _detect_language(self, content: str) -> str:
        """Detect the primary language of the document"""
        # Simple language detection based on character patterns
        if re.search(r'[\u4e00-\u9fff]', content):  # Chinese characters
            return "chinese"
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', content):  # Japanese
            return "japanese"
        elif re.search(r'[\u0400-\u04ff]', content):  # Cyrillic
            return "russian"
        elif re.search(r'[\u0590-\u05ff]', content):  # Hebrew
            return "hebrew"
        elif re.search(r'[\u0600-\u06ff]', content):  # Arabic
            return "arabic"
        else:
            # Check for common words in different languages
            english_words = len(re.findall(r'\b(the|and|or|of|to|in|for|with|by)\b', content, re.IGNORECASE))
            french_words = len(re.findall(r'\b(le|la|les|et|ou|de|du|des|pour|avec)\b', content, re.IGNORECASE))
            german_words = len(re.findall(r'\b(der|die|das|und|oder|von|zu|mit|für)\b', content, re.IGNORECASE))
            spanish_words = len(re.findall(r'\b(el|la|los|las|y|o|de|del|para|con)\b', content, re.IGNORECASE))
            
            max_words = max(english_words, french_words, german_words, spanish_words)
            if max_words == english_words and english_words > 5:
                return "english"
            elif max_words == french_words and french_words > 5:
                return "french"
            elif max_words == german_words and german_words > 5:
                return "german"
            elif max_words == spanish_words and spanish_words > 5:
                return "spanish"
        
        return "unknown"
    
    def _assess_complexity(self, content: str) -> str:
        """Assess the structural complexity of the document"""
        score = 0
        
        # Header levels
        header_levels = len(set(re.findall(r'^(#{1,6})', content, re.MULTILINE)))
        score += header_levels * 2
        
        # Tables
        table_count = len(re.findall(r'\|.*\|.*\|', content))
        score += min(table_count, 10)
        
        # Lists
        list_count = len(re.findall(r'^\s*[-*+\d]+\.?\s+', content, re.MULTILINE))
        score += min(list_count // 5, 5)
        
        # Special formatting
        if re.search(r'\*\*.*\*\*|__.*__|```', content):
            score += 3
        
        # Length
        word_count = len(content.split())
        if word_count > 5000:
            score += 5
        elif word_count > 1000:
            score += 2
        
        if score >= 15:
            return "high"
        elif score >= 8:
            return "medium"
        else:
            return "low"
    
    def generate_patterns(self) -> Dict[str, DocumentPattern]:
        """Generate extraction patterns for all document types found"""
        if not self.data_dir.exists():
            print(f"Directory {self.data_dir} does not exist")
            return {}
        
        # Analyze all markdown files
        analyses = []
        for md_file in self.data_dir.glob("*.md"):
            analysis = self.analyze_document(md_file)
            if "error" not in analysis:
                analyses.append(analysis)
        
        # Group by document type
        type_groups = defaultdict(list)
        for analysis in analyses:
            type_groups[analysis["document_type"]].append(analysis)
        
        # Generate patterns for each type
        patterns = {}
        for doc_type, docs in type_groups.items():
            pattern = self._create_pattern_for_type(doc_type, docs)
            patterns[doc_type] = pattern
        
        return patterns
    
    def _create_pattern_for_type(self, doc_type: str, docs: List[Dict]) -> DocumentPattern:
        """Create extraction pattern for a specific document type"""
        
        # Collect common patterns across documents of this type
        all_headers = []
        all_sections = []
        all_metadata = []
        all_elements = []
        
        for doc in docs:
            all_headers.extend([h["pattern"] for h in doc["headers"]])
            all_sections.extend([s["type"] for s in doc["sections"]])
            all_metadata.extend(list(doc["metadata"].keys()))
            all_elements.extend(doc["special_elements"])
        
        # Find most common patterns
        header_patterns = list(set(all_headers))[:10]  # Top 10 unique header patterns
        section_patterns = list(set(all_sections))
        metadata_patterns = list(set(all_metadata))
        content_indicators = list(set(all_elements))
        
        # Define special features based on document type
        special_features = self._get_special_features_for_type(doc_type, docs)
        
        # Create description
        description = self._generate_description(doc_type, docs)
        
        return DocumentPattern(
            name=doc_type,
            description=description,
            header_patterns=header_patterns,
            section_patterns=section_patterns,
            metadata_patterns=metadata_patterns,
            content_indicators=content_indicators,
            special_features=special_features
        )
    
    def _get_special_features_for_type(self, doc_type: str, docs: List[Dict]) -> List[str]:
        """Get special features for a document type"""
        features = []
        
        if doc_type == "financial":
            features = [
                "account_numbers", "transaction_dates", "amounts", 
                "reference_numbers", "bank_details", "payment_methods"
            ]
        elif doc_type == "legal":
            features = [
                "clause_numbers", "definitions", "terms_conditions",
                "signatures", "dates", "legal_references"
            ]
        elif doc_type == "corporate_policy":
            features = [
                "policy_sections", "compliance_requirements", "procedures",
                "responsibilities", "contact_information", "effective_dates"
            ]
        elif doc_type == "travel_tourism":
            features = [
                "locations", "dates", "contact_details", "schedules",
                "prices", "booking_references", "addresses"
            ]
        elif doc_type == "technical_manual":
            features = [
                "step_by_step_instructions", "diagrams", "troubleshooting",
                "specifications", "safety_warnings", "examples"
            ]
        elif doc_type == "medical_health":
            features = [
                "patient_info", "medical_terms", "procedures",
                "dates", "contact_details", "instructions"
            ]
        elif doc_type == "government_official":
            features = [
                "form_fields", "official_numbers", "dates",
                "requirements", "contact_information", "procedures"
            ]
        elif doc_type == "form":
            features = [
                "form_fields", "checkboxes", "signatures",
                "dates", "reference_numbers", "instructions"
            ]
        else:
            features = ["general_content", "headers", "paragraphs"]
        
        return features
    
    def _generate_description(self, doc_type: str, docs: List[Dict]) -> str:
        """Generate description for document type"""
        doc_count = len(docs)
        languages = set(doc["language"] for doc in docs)
        complexities = set(doc["structure_complexity"] for doc in docs)
        
        description = f"Pattern for {doc_type} documents ({doc_count} samples). "
        description += f"Languages: {', '.join(languages)}. "
        description += f"Complexity levels: {', '.join(complexities)}."
        
        return description
    
    def save_patterns(self, patterns: Dict[str, DocumentPattern], output_file: str = "extraction_patterns.py"):
        """Save patterns to a Python file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\n')
            f.write('Generated extraction patterns for markdown documents in data/misc folder.\n')
            f.write('This file contains patterns for different document types found in the dataset.\n')
            f.write('"""\n\n')
            f.write('from typing import Dict, List, Any\n')
            f.write('import re\n\n')
            
            # Write pattern definitions
            f.write('EXTRACTION_PATTERNS = {\n')
            for doc_type, pattern in patterns.items():
                f.write(f'    "{doc_type}": {{\n')
                f.write(f'        "name": "{pattern.name}",\n')
                f.write(f'        "description": "{pattern.description}",\n')
                f.write(f'        "header_patterns": {pattern.header_patterns},\n')
                f.write(f'        "section_patterns": {pattern.section_patterns},\n')
                f.write(f'        "metadata_patterns": {pattern.metadata_patterns},\n')
                f.write(f'        "content_indicators": {pattern.content_indicators},\n')
                f.write(f'        "special_features": {pattern.special_features}\n')
                f.write('    },\n')
            f.write('}\n\n')
            
            # Write extraction functions
            self._write_extraction_functions(f)
    
    def _write_extraction_functions(self, f):
        """Write extraction functions to the output file"""
        f.write('''
def extract_headers(content: str, doc_type: str = "general") -> List[Dict]:
    """Extract headers based on document type patterns"""
    headers = []
    
    # Markdown headers
    for match in re.finditer(r'^(#{1,6})\\s+(.+)$', content, re.MULTILINE):
        headers.append({
            "type": "markdown",
            "level": len(match.group(1)),
            "text": match.group(2).strip(),
            "line": content[:match.start()].count('\\n') + 1
        })
    
    # Underlined headers
    lines = content.split('\\n')
    for i, line in enumerate(lines[:-1]):
        next_line = lines[i + 1]
        if re.match(r'^=+$', next_line.strip()) and line.strip():
            headers.append({
                "type": "underlined_h1",
                "level": 1,
                "text": line.strip(),
                "line": i + 1
            })
        elif re.match(r'^-+$', next_line.strip()) and line.strip():
            headers.append({
                "type": "underlined_h2",
                "level": 2,
                "text": line.strip(),
                "line": i + 1
            })
    
    return headers

def extract_metadata(content: str, doc_type: str = "general") -> Dict[str, Any]:
    """Extract metadata based on document type"""
    metadata = {}
    
    # Common metadata patterns
    dates = re.findall(r'\\b(\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}|\\d{1,2}[-/]\\d{1,2}[-/]\\d{4})', content)
    if dates:
        metadata["dates"] = dates
    
    emails = re.findall(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', content)
    if emails:
        metadata["emails"] = emails
    
    urls = re.findall(r'https?://[^\\s<>"]+', content)
    if urls:
        metadata["urls"] = urls
    
    phones = re.findall(r'[\\+]?[\\d\\s\\-\\(\\)]{10,}', content)
    if phones:
        metadata["phones"] = [p.strip() for p in phones]
    
    # Document type specific metadata
    if doc_type == "financial":
        account_nums = re.findall(r'\\b\\d{3}-\\d{6}-\\d{3}|\\b\\d{10,}\\b', content)
        if account_nums:
            metadata["account_numbers"] = account_nums
        
        amounts = re.findall(r'[A-Z]{3}\\s*\\d+[,.]\\d{2}|\\$\\s*\\d+[,.]\\d{2}', content)
        if amounts:
            metadata["amounts"] = amounts
    
    elif doc_type == "legal":
        clauses = re.findall(r'\\b(Article|Section|Clause)\\s+\\d+', content, re.IGNORECASE)
        if clauses:
            metadata["legal_references"] = clauses
    
    elif doc_type == "travel_tourism":
        locations = re.findall(r'\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*(?:\\s+\\d{5})?\\b', content)
        if locations:
            metadata["locations"] = locations[:10]  # Limit to avoid noise
    
    return metadata

def extract_sections(content: str, doc_type: str = "general") -> List[Dict]:
    """Extract sections based on document type patterns"""
    sections = []
    
    # Numbered sections
    numbered = re.finditer(r'^\\s*(\\d+\\.?\\d*\\.?\\d*)\\s+(.+)$', content, re.MULTILINE)
    for match in numbered:
        sections.append({
            "type": "numbered",
            "number": match.group(1),
            "title": match.group(2).strip(),
            "line": content[:match.start()].count('\\n') + 1
        })
    
    # Lettered sections
    lettered = re.finditer(r'^\\s*([A-Za-z]\\.)\\s+(.+)$', content, re.MULTILINE)
    for match in lettered:
        sections.append({
            "type": "lettered",
            "letter": match.group(1),
            "title": match.group(2).strip(),
            "line": content[:match.start()].count('\\n') + 1
        })
    
    # Document type specific sections
    if doc_type == "corporate_policy":
        policy_sections = re.finditer(r'^\\s*(Policy|Procedure|Requirement)\\s*:?\\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
        for match in policy_sections:
            sections.append({
                "type": "policy_section",
                "category": match.group(1).lower(),
                "title": match.group(2).strip(),
                "line": content[:match.start()].count('\\n') + 1
            })
    
    return sections

def extract_content_by_type(content: str, doc_type: str = "general") -> Dict[str, Any]:
    """Extract content based on document type"""
    result = {
        "headers": extract_headers(content, doc_type),
        "metadata": extract_metadata(content, doc_type),
        "sections": extract_sections(content, doc_type),
        "document_type": doc_type
    }
    
    # Add type-specific extractions
    if doc_type in EXTRACTION_PATTERNS:
        pattern = EXTRACTION_PATTERNS[doc_type]
        result["pattern_info"] = {
            "name": pattern["name"],
            "description": pattern["description"],
            "special_features": pattern["special_features"]
        }
    
    return result

def classify_document_type(content: str, filename: str = "") -> str:
    """Classify document type based on content and filename"""
    content_lower = content.lower()
    filename_lower = filename.lower()
    
    # Check each pattern type
    for doc_type, pattern in EXTRACTION_PATTERNS.items():
        score = 0
        
        # Check for content indicators
        for indicator in pattern["content_indicators"]:
            if indicator.replace("_", " ") in content_lower:
                score += 1
        
        # Check special features
        for feature in pattern["special_features"]:
            if feature.replace("_", " ") in content_lower:
                score += 2
        
        # Filename matching
        if doc_type.replace("_", " ") in filename_lower:
            score += 3
        
        # Return first type with significant score
        if score >= 3:
            return doc_type
    
    return "general"
''')

def main():
    """Main function to generate extraction patterns"""
    generator = PatternGenerator()
    
    print("Analyzing markdown files in data/misc folder...")
    patterns = generator.generate_patterns()
    
    if not patterns:
        print("No patterns generated. Check if data/misc folder exists and contains .md files.")
        return
    
    print(f"Generated patterns for {len(patterns)} document types:")
    for doc_type, pattern in patterns.items():
        print(f"  - {doc_type}: {pattern.description}")
    
    # Save patterns to file
    output_file = "extraction_patterns.py"
    generator.save_patterns(patterns, output_file)
    print(f"\nPatterns saved to {output_file}")
    
    # Generate summary report
    with open("pattern_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write("# Document Pattern Analysis Report\n\n")
        f.write(f"Generated from {len(patterns)} document types found in data/misc folder.\n\n")
        
        for doc_type, pattern in patterns.items():
            f.write(f"## {doc_type.replace('_', ' ').title()}\n\n")
            f.write(f"**Description:** {pattern.description}\n\n")
            
            if pattern.header_patterns:
                f.write("**Header Patterns:**\n")
                for hp in pattern.header_patterns[:5]:  # Show first 5
                    f.write(f"- `{hp}`\n")
                f.write("\n")
            
            if pattern.section_patterns:
                f.write("**Section Types:**\n")
                for sp in pattern.section_patterns:
                    f.write(f"- {sp}\n")
                f.write("\n")
            
            if pattern.metadata_patterns:
                f.write("**Metadata Types:**\n")
                for mp in pattern.metadata_patterns:
                    f.write(f"- {mp}\n")
                f.write("\n")
            
            if pattern.special_features:
                f.write("**Special Features:**\n")
                for sf in pattern.special_features:
                    f.write(f"- {sf}\n")
                f.write("\n")
            
            f.write("---\n\n")
    
    print("Analysis report saved to pattern_analysis_report.md")

if __name__ == "__main__":
    main()