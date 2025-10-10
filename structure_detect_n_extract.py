#!/usr/bin/env python3

import re
from collections import defaultdict, Counter

# Regular expression pattern for markdown structure extraction
STRUCTURE_PATTERN = r'''
(?P<schedule>\*\*Schedule\s+\d+[A-Z]*[^*]*\*\*)|\
(?P<part>\*\*Part\s+\d+\*\*\s*\n\s*\*\*[^*]+\*\*)|\
(?P<division>\*\*Division\s+\d+[^*]*\*\*)|\
(?P<subdivision>\*\*Subdivision\s+\d+[^*]*\*\*)|\
(?P<section>\*\*\d+\.\s+[^*]+\*\*)
'''

def extract_structure(markdown_text):
    """Extract hierarchical structure from markdown text"""
    pattern = re.compile(STRUCTURE_PATTERN, re.VERBOSE | re.MULTILINE)
    
    structure = []
    current_schedule = None
    current_part = None
    current_division = None
    current_subdivision = None
    
    matches = list(pattern.finditer(markdown_text))
    
    for i, match in enumerate(matches):
        groups = match.groupdict()
        
        # Extract content between current match and next match
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start_pos:end_pos].strip()
        
        # Clean up content (remove extra whitespace and empty lines)
        if content:
            content_lines = [line.strip() for line in content.split('\n') if line.strip()]
            content = '\n'.join(content_lines)
        
        if groups['schedule']:
            # Extract schedule number and title
            schedule_match = re.search(r'Schedule\s+(\d+[A-Z]*)(?:—(.+))?', groups['schedule'])
            if schedule_match:
                current_schedule = {
                    'type': 'schedule',
                    'number': schedule_match.group(1),
                    'title': schedule_match.group(2).strip() if schedule_match.group(2) else '',
                    'parts': [],
                    'sections': [],
                    'content': content
                }
                structure.append(current_schedule)
                current_part = None
                current_division = None
                current_subdivision = None
        
        elif groups['part']:
            # Extract part number and title
            part_match = re.search(r'Part\s+(\d+)\*\*\s*\n\s*\*\*([^*]+)', groups['part'])
            if part_match:
                current_part = {
                    'type': 'part',
                    'number': part_match.group(1),
                    'title': part_match.group(2).strip(),
                    'divisions': [],
                    'content': content
                }
                if current_schedule:
                    current_schedule['parts'].append(current_part)
                else:
                    structure.append(current_part)
                current_division = None
                current_subdivision = None
        
        elif groups['division']:
            # Extract division number and title
            div_match = re.search(r'Division\s+(\d+)(?:—(.+))?', groups['division'])
            if div_match and current_part:
                current_division = {
                    'type': 'division',
                    'number': div_match.group(1),
                    'title': div_match.group(2).strip() if div_match.group(2) else '',
                    'subdivisions': [],
                    'sections': [],
                    'content': content
                }
                current_part['divisions'].append(current_division)
                current_subdivision = None
        
        elif groups['subdivision']:
            # Extract subdivision number and title
            subdiv_match = re.search(r'Subdivision\s+(\d+)(?:—(.+))?', groups['subdivision'])
            if subdiv_match and current_division:
                current_subdivision = {
                    'type': 'subdivision',
                    'number': subdiv_match.group(1),
                    'title': subdiv_match.group(2).strip() if subdiv_match.group(2) else '',
                    'sections': [],
                    'content': content
                }
                current_division['subdivisions'].append(current_subdivision)
        
        elif groups['section']:
            # Extract section number and title
            sec_match = re.search(r'(\d+)\.\s+(.+)', groups['section'])
            if sec_match:
                section = {
                    'type': 'section',
                    'number': sec_match.group(1),
                    'title': sec_match.group(2).strip(),
                    'content': content
                }
                
                # Add section to appropriate container
                if current_subdivision:
                    current_subdivision['sections'].append(section)
                elif current_division:
                    current_division['sections'].append(section)
                elif current_part:
                    if 'sections' not in current_part:
                        current_part['sections'] = []
                    current_part['sections'].append(section)
                elif current_schedule:
                    current_schedule['sections'].append(section)
    
    return structure

def detect_and_extract(markdown_text):
    """Alternative extraction method - returns same format as extract_structure()"""
    return extract_structure(markdown_text)

class StructureDetector:
    def __init__(self):
        self.patterns = {}
        self.hierarchy_levels = []
    
    def detect_structure(self, markdown_text):
        """Detect structure patterns in markdown text"""
        # Find all potential headers
        headers = self._find_headers(markdown_text)
        
        # Group numbered patterns under markdown headers
        grouped_headers = self._group_under_headers(headers)
        
        # Analyze patterns
        patterns = self._analyze_patterns(headers)
        
        # Build hierarchy
        hierarchy = self._build_hierarchy(patterns, grouped_headers)
        
        return {
            'patterns': patterns,
            'hierarchy': hierarchy,
            'grouped_headers': grouped_headers,
            'extractor': self._create_extractor(patterns)
        }
    
    def _find_headers(self, text):
        """Find all potential header patterns"""
        headers = []
        
        # Markdown headers (#, ##, ###, etc.)
        md_headers = re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
        md_count = 0
        for match in md_headers:
            headers.append({
                'type': 'markdown',
                'level': len(match.group(1)),
                'text': match.group(2).strip(),
                'position': match.start(),
                'line': text[:match.start()].count('\n') + 1
            })
            md_count += 1
        print(f"DEBUG: Found {md_count} markdown headers")
        
        # Bold patterns (**text**)
        bold_patterns = re.finditer(r'\*\*([^*]+)\*\*', text)
        bold_count = 0
        for match in bold_patterns:
            headers.append({
                'type': 'bold',
                'text': match.group(1).strip(),
                'position': match.start(),
                'line': text[:match.start()].count('\n') + 1
            })
            bold_count += 1
        print(f"DEBUG: Found {bold_count} bold patterns")
        
        # Numbered patterns (1., 2.1, etc.)
        numbered = re.finditer(r'^(\d+(?:\.\d+)*)\.\s+(.+)$', text, re.MULTILINE)
        num_count = 0
        for match in numbered:
            headers.append({
                'type': 'numbered',
                'number': match.group(1),
                'text': match.group(2).strip(),
                'position': match.start(),
                'line': text[:match.start()].count('\n') + 1
            })
            num_count += 1
        print(f"DEBUG: Found {num_count} numbered patterns")
        
        return sorted(headers, key=lambda x: x['position'])
    
    def _analyze_patterns(self, headers):
        """Analyze header patterns to identify structure types"""
        patterns = {}
        
        # Analyze bold headers for common legal/document patterns
        bold_headers = [h for h in headers if h['type'] == 'bold']
        print(f"DEBUG: Analyzing {len(bold_headers)} bold headers")
        
        for header in bold_headers:
            text = header['text']
            
            # Schedule patterns
            if re.match(r'Schedule\s+\d+[A-Z]*', text, re.IGNORECASE):
                patterns['schedule'] = r'\*\*Schedule\s+(\d+[A-Z]*)(?:[—\-:](.+))?\*\*'
                print(f"DEBUG: Found schedule pattern: {text}")
            
            # Part patterns
            elif re.match(r'Part\s+\d+', text, re.IGNORECASE):
                patterns['part'] = r'\*\*Part\s+(\d+)(?:[—\-:](.+))?\*\*'
                print(f"DEBUG: Found part pattern: {text}")
            
            # Division patterns
            elif re.match(r'Division\s+\d+', text, re.IGNORECASE):
                patterns['division'] = r'\*\*Division\s+(\d+)(?:[—\-:](.+))?\*\*'
                print(f"DEBUG: Found division pattern: {text}")
            
            # Section patterns (numbered)
            elif re.match(r'\d+\.\s+', text):
                patterns['section'] = r'\*\*(\d+)\.\s+(.+)\*\*'
                print(f"DEBUG: Found section pattern: {text}")
            
            # Chapter patterns
            elif re.match(r'Chapter\s+\d+', text, re.IGNORECASE):
                patterns['chapter'] = r'\*\*Chapter\s+(\d+)(?:[—\-:](.+))?\*\*'
                print(f"DEBUG: Found chapter pattern: {text}")
            
            # Article patterns
            elif re.match(r'Article\s+\d+', text, re.IGNORECASE):
                patterns['article'] = r'\*\*Article\s+(\d+)(?:[—\-:](.+))?\*\*'
                print(f"DEBUG: Found article pattern: {text}")
        
        # Analyze markdown headers
        md_headers = [h for h in headers if h['type'] == 'markdown']
        if md_headers:
            level_counts = Counter(h['level'] for h in md_headers)
            patterns['markdown'] = {f'h{level}': f'^#{{{level}}}\\s+(.+)$' 
                                  for level in level_counts.keys()}
            print(f"DEBUG: Found markdown levels: {list(level_counts.keys())}")
        
        return patterns
    
    def _build_hierarchy(self, patterns, grouped_headers=None):
        """Build hierarchy based on detected patterns and grouped headers"""
        hierarchy = []
        
        # Add prelims if it exists
        if grouped_headers:
            for group in grouped_headers:
                if group.get('type') == 'prelims':
                    hierarchy.append('prelims')
                    break
        
        # Add markdown headers with their grouped numbered items
        if grouped_headers:
            md_levels = set()
            for group in grouped_headers:
                if group.get('type') == 'header_group':
                    level = group['header']['level']
                    md_levels.add(level)
            
            for level in sorted(md_levels):
                hierarchy.append(f'markdown_h{level}')
                hierarchy.append(f'numbered_under_h{level}')
        
        # Legal document hierarchy
        legal_order = ['schedule', 'part', 'division', 'subdivision', 'section']
        for item in legal_order:
            if item in patterns:
                hierarchy.append(item)
        
        # Academic/book hierarchy
        academic_order = ['chapter', 'section', 'subsection']
        for item in academic_order:
            if item in patterns:
                hierarchy.append(item)
        
        # Article-based hierarchy
        if 'article' in patterns:
            hierarchy.append('article')
        
        return hierarchy
    
    def _group_under_headers(self, headers):
        """Group numbered patterns under closest markdown headers"""
        md_headers = [h for h in headers if h['type'] == 'markdown']
        numbered = [h for h in headers if h['type'] == 'numbered']
        
        grouped = []
        
        # Find first structural element (markdown header or numbered item)
        first_structure_pos = None
        if md_headers or numbered:
            all_structural = md_headers + numbered
            first_structure_pos = min(h['position'] for h in all_structural)
        
        # Add prelims group if there's content before first structure
        if first_structure_pos is not None and first_structure_pos > 0:
            grouped.append({
                'type': 'prelims',
                'name': 'prelims',
                'position': 0,
                'end_position': first_structure_pos
            })
        
        for num_item in numbered:
            # Find closest preceding markdown header
            closest_header = None
            for md_header in reversed(md_headers):
                if md_header['position'] < num_item['position']:
                    closest_header = md_header
                    break
            
            if closest_header:
                # Check if header group already exists
                existing_group = None
                for group in grouped:
                    if (group['type'] == 'header_group' and 
                        group['header']['position'] == closest_header['position']):
                        existing_group = group
                        break
                
                if existing_group:
                    existing_group['items'].append(num_item)
                else:
                    grouped.append({
                        'type': 'header_group',
                        'header': closest_header,
                        'items': [num_item]
                    })
            else:
                # No preceding header, add as standalone
                grouped.append(num_item)
        
        return grouped
    
    def _create_extractor(self, patterns):
        """Create extraction function based on detected patterns"""
        def extract_structure_dynamic(markdown_text):
            # Fallback to main extract_structure if no specific patterns
            return extract_structure(markdown_text)
        
        return extract_structure_dynamic

# Compact regex for direct extraction (single pattern)
COMPACT_PATTERN = r'\*\*(?:Schedule\s+(\d+[A-Z]*)(?:—([^*]+))?|Part\s+(\d+)\*\*\s*\n\s*\*\*([^*]+)|Division\s+(\d+)(?:—([^*]+))?|Subdivision\s+(\d+)(?:—([^*]+))?|(\d+)\.\s+([^*]+))\*\*'