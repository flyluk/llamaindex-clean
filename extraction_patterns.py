#!/usr/bin/env python3
"""
Generated extraction patterns for markdown documents in data/misc folder.
This file contains patterns for different document types found in the dataset.
"""

from typing import Dict, List, Any
import re

EXTRACTION_PATTERNS = {
    "corporate_policy": {
        "name": "corporate_policy",
        "description": "Pattern for corporate_policy documents (2 samples). Languages: english, chinese. Complexity levels: low.",
        "header_patterns": ['## PERSONAL-IN-CONFIDENCE 私人密件', '## Health Declaration of COVID-19 Infection 2019 冠狀病毒 感染 健康申報表', '## CONFLICT OF INTEREST'],
        "section_patterns": ['numbered_sections'],
        "metadata_patterns": ['phones'],
        "content_indicators": ['images', 'numbered_lists', 'code_blocks', 'emphasis'],
        "special_features": ['policy_sections', 'compliance_requirements', 'procedures', 'responsibilities', 'contact_information', 'effective_dates']
    },
    "financial": {
        "name": "financial",
        "description": "Pattern for financial documents (14 samples). Languages: english, chinese. Complexity levels: medium, high, low.",
        "header_patterns": ['## INFORMATION ABOUT OUR EXECUTIVE OFFICERS', '## Put the sail on what would be/is the upwind side of the board when the board is pointing in the direction we want to start in', '## Over the head back foot-strap and mast', '## 1. Let go of the sail.', '## CONSOLIDATED STATEMENTS OF CASH FLOWS', '## Watch the pros', '## II 申請人資料 Applicant Information', '## Planing upwind', '## We act with speed and purpose.', '## RECENTLY ISSUED ACCOUNTING STANDARDS'],
        "section_patterns": ['table_of_contents', 'numbered_sections', 'lettered_sections'],
        "metadata_patterns": ['reference_numbers', 'dates', 'phones', 'emails', 'urls'],
        "content_indicators": ['images', 'tables', 'bullet_lists', 'numbered_lists', 'emphasis', 'code_blocks'],
        "special_features": ['account_numbers', 'transaction_dates', 'amounts', 'reference_numbers', 'bank_details', 'payment_methods']
    },
    "travel_tourism": {
        "name": "travel_tourism",
        "description": "Pattern for travel_tourism documents (1 samples). Languages: english. Complexity levels: low.",
        "header_patterns": ['## Cabin Baggage Allowance', '## Travel Plan', '## Departure', '## Discover with Turkish Airlines Blog', '## Cabin Baggage Dimensions', '## Getting To The Airport', '## Airport Security Check', '## General Statement', '## Baggage Delivery', '## Boarding'],
        "section_patterns": ['numbered_sections'],
        "metadata_patterns": [],
        "content_indicators": ['bullet_lists', 'images'],
        "special_features": ['locations', 'dates', 'contact_details', 'schedules', 'prices', 'booking_references', 'addresses']
    },
    "general": {
        "name": "general",
        "description": "Pattern for general documents (2 samples). Languages: chinese. Complexity levels: high, low.",
        "header_patterns": ['## TransLantau 50', '## Results 2017 - Overall ranking'],
        "section_patterns": ['numbered_sections'],
        "metadata_patterns": ['phones', 'reference_numbers', 'urls'],
        "content_indicators": ['images', 'tables', 'numbered_lists', 'emphasis', 'code_blocks'],
        "special_features": ['general_content', 'headers', 'paragraphs']
    },
}


def extract_headers(content: str, doc_type: str = "general") -> List[Dict]:
    """Extract headers based on document type patterns"""
    headers = []
    
    # Markdown headers
    for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
        headers.append({
            "type": "markdown",
            "level": len(match.group(1)),
            "text": match.group(2).strip(),
            "line": content[:match.start()].count('\n') + 1
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
    dates = re.findall(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', content)
    if dates:
        metadata["dates"] = dates
    
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
    if emails:
        metadata["emails"] = emails
    
    urls = re.findall(r'https?://[^\s<>"]+', content)
    if urls:
        metadata["urls"] = urls
    
    phones = re.findall(r'[\+]?[\d\s\-\(\)]{10,}', content)
    if phones:
        metadata["phones"] = [p.strip() for p in phones]
    
    # Document type specific metadata
    if doc_type == "financial":
        account_nums = re.findall(r'\b\d{3}-\d{6}-\d{3}|\b\d{10,}\b', content)
        if account_nums:
            metadata["account_numbers"] = account_nums
        
        amounts = re.findall(r'[A-Z]{3}\s*\d+[,.]\d{2}|\$\s*\d+[,.]\d{2}', content)
        if amounts:
            metadata["amounts"] = amounts
    
    elif doc_type == "legal":
        clauses = re.findall(r'\b(Article|Section|Clause)\s+\d+', content, re.IGNORECASE)
        if clauses:
            metadata["legal_references"] = clauses
    
    elif doc_type == "travel_tourism":
        locations = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+\d{5})?\b', content)
        if locations:
            metadata["locations"] = locations[:10]  # Limit to avoid noise
    
    return metadata

def extract_sections(content: str, doc_type: str = "general") -> List[Dict]:
    """Extract sections based on document type patterns"""
    sections = []
    
    # Numbered sections
    numbered = re.finditer(r'^\s*(\d+\.?\d*\.?\d*)\s+(.+)$', content, re.MULTILINE)
    for match in numbered:
        sections.append({
            "type": "numbered",
            "number": match.group(1),
            "title": match.group(2).strip(),
            "line": content[:match.start()].count('\n') + 1
        })
    
    # Lettered sections
    lettered = re.finditer(r'^\s*([A-Za-z]\.)\s+(.+)$', content, re.MULTILINE)
    for match in lettered:
        sections.append({
            "type": "lettered",
            "letter": match.group(1),
            "title": match.group(2).strip(),
            "line": content[:match.start()].count('\n') + 1
        })
    
    # Document type specific sections
    if doc_type == "corporate_policy":
        policy_sections = re.finditer(r'^\s*(Policy|Procedure|Requirement)\s*:?\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
        for match in policy_sections:
            sections.append({
                "type": "policy_section",
                "category": match.group(1).lower(),
                "title": match.group(2).strip(),
                "line": content[:match.start()].count('\n') + 1
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
