"""
Text processing and manipulation tools.
"""

import re
import json
import base64
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, unquote

from ..base import BaseTool, ToolCategory, ToolMetadata, ToolParameter, ToolPermission


class TextAnalysisTool(BaseTool):
    """Tool for analyzing text content."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="text_analysis",
            description="Analyze text for statistics, readability, and patterns",
            category=ToolCategory.UTILITY,
            version="1.0.0",
            author="System",
            permissions=[],
            examples=[
                'text_analysis(text="Hello world! This is a test.")',
                'text_analysis(text="Sample text", include_readability=true)',
                'text_analysis(text="Text with patterns", find_patterns=["email", "url"])'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="str",
                description="Text to analyze",
                required=True
            ),
            ToolParameter(
                name="include_readability",
                type="bool",
                description="Include readability metrics",
                required=False,
                default=False
            ),
            ToolParameter(
                name="find_patterns",
                type="list",
                description="Patterns to find (email, url, phone, etc.)",
                required=False,
                default=[]
            )
        ]
    
    async def _execute(
        self,
        text: str,
        include_readability: bool = False,
        find_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze text content."""
        if find_patterns is None:
            find_patterns = []
        
        # Basic statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        analysis = {
            "basic_stats": {
                "character_count": len(text),
                "character_count_no_spaces": len(text.replace(' ', '')),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "average_words_per_sentence": len(words) / len(sentences) if sentences else 0,
                "average_sentences_per_paragraph": len(sentences) / len(paragraphs) if paragraphs else 0
            },
            "character_distribution": self._get_character_distribution(text),
            "word_frequency": self._get_word_frequency(words)
        }
        
        if include_readability:
            analysis["readability"] = self._calculate_readability(text, words, sentences)
        
        if find_patterns:
            analysis["patterns"] = self._find_patterns(text, find_patterns)
        
        return analysis
    
    def _get_character_distribution(self, text: str) -> Dict[str, int]:
        """Get character type distribution."""
        stats = {
            "letters": 0,
            "digits": 0,
            "spaces": 0,
            "punctuation": 0,
            "other": 0
        }
        
        for char in text:
            if char.isalpha():
                stats["letters"] += 1
            elif char.isdigit():
                stats["digits"] += 1
            elif char.isspace():
                stats["spaces"] += 1
            elif char in ".,!?;:\"'()[]{}":
                stats["punctuation"] += 1
            else:
                stats["other"] += 1
        
        return stats
    
    def _get_word_frequency(self, words: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top word frequencies."""
        # Normalize words
        normalized_words = [word.lower().strip('.,!?;:"()[]{}') for word in words]
        normalized_words = [word for word in normalized_words if word]
        
        # Count frequencies
        freq_map = {}
        for word in normalized_words:
            freq_map[word] = freq_map.get(word, 0) + 1
        
        # Sort and get top N
        sorted_words = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"word": word, "count": count, "percentage": (count / len(normalized_words)) * 100}
            for word, count in sorted_words[:top_n]
        ]
    
    def _calculate_readability(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, Any]:
        """Calculate basic readability metrics."""
        if not words or not sentences:
            return {"error": "Not enough text for readability analysis"}
        
        # Count syllables (approximate)
        syllable_count = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease Score
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllable_count / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        return {
            "flesch_reading_ease": max(0, min(100, flesch_score)),
            "flesch_kincaid_grade": max(0, fk_grade),
            "avg_sentence_length": avg_sentence_length,
            "avg_syllables_per_word": avg_syllables_per_word,
            "total_syllables": syllable_count
        }
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word."""
        word = word.lower().strip('.,!?;:"()[]{}')
        if not word:
            return 0
        
        # Simple syllable counting heuristic
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _find_patterns(self, text: str, patterns: List[str]) -> Dict[str, List[str]]:
        """Find common patterns in text."""
        pattern_regexes = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://[^\s<>"\']+',
            "phone": r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "time": r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "social_security": r'\b\d{3}-\d{2}-\d{4}\b'
        }
        
        results = {}
        for pattern_name in patterns:
            if pattern_name in pattern_regexes:
                matches = re.findall(pattern_regexes[pattern_name], text)
                results[pattern_name] = matches
            else:
                results[pattern_name] = f"Unknown pattern: {pattern_name}"
        
        return results


class TextTransformTool(BaseTool):
    """Tool for transforming and manipulating text."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="text_transform",
            description="Transform text with various operations (case, encoding, formatting)",
            category=ToolCategory.UTILITY,
            version="1.0.0",
            author="System",
            permissions=[],
            examples=[
                'text_transform(text="Hello World", operation="upper")',
                'text_transform(text="some text", operation="title_case")',
                'text_transform(text="Hello World", operation="base64_encode")',
                'text_transform(text="SGVsbG8gV29ybGQ=", operation="base64_decode")'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="str",
                description="Text to transform",
                required=True
            ),
            ToolParameter(
                name="operation",
                type="str",
                description="Transformation operation to perform",
                required=True,
                choices=[
                    "upper", "lower", "title_case", "sentence_case", "camel_case", "snake_case",
                    "reverse", "remove_whitespace", "normalize_whitespace", "remove_punctuation",
                    "base64_encode", "base64_decode", "url_encode", "url_decode",
                    "json_format", "word_wrap", "remove_duplicates", "sort_lines"
                ]
            ),
            ToolParameter(
                name="options",
                type="dict",
                description="Additional options for the operation",
                required=False,
                default={}
            )
        ]
    
    async def _execute(
        self,
        text: str,
        operation: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Transform text using the specified operation."""
        if options is None:
            options = {}
        
        try:
            if operation == "upper":
                result = text.upper()
            elif operation == "lower":
                result = text.lower()
            elif operation == "title_case":
                result = text.title()
            elif operation == "sentence_case":
                result = text.capitalize()
            elif operation == "camel_case":
                words = re.split(r'[^a-zA-Z0-9]', text)
                result = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
            elif operation == "snake_case":
                result = re.sub(r'[^a-zA-Z0-9]', '_', text).lower()
            elif operation == "reverse":
                result = text[::-1]
            elif operation == "remove_whitespace":
                result = re.sub(r'\s+', '', text)
            elif operation == "normalize_whitespace":
                result = re.sub(r'\s+', ' ', text).strip()
            elif operation == "remove_punctuation":
                result = re.sub(r'[^\w\s]', '', text)
            elif operation == "base64_encode":
                result = base64.b64encode(text.encode('utf-8')).decode('utf-8')
            elif operation == "base64_decode":
                result = base64.b64decode(text.encode('utf-8')).decode('utf-8')
            elif operation == "url_encode":
                result = quote(text)
            elif operation == "url_decode":
                result = unquote(text)
            elif operation == "json_format":
                try:
                    parsed = json.loads(text)
                    result = json.dumps(parsed, indent=options.get('indent', 2))
                except json.JSONDecodeError as e:
                    return {"success": False, "error": f"Invalid JSON: {e}"}
            elif operation == "word_wrap":
                width = options.get('width', 80)
                words = text.split()
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + len(current_line) > width:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                        else:
                            lines.append(word)
                    else:
                        current_line.append(word)
                        current_length += len(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                result = '\n'.join(lines)
            elif operation == "remove_duplicates":
                lines = text.split('\n')
                seen = set()
                unique_lines = []
                for line in lines:
                    if line not in seen:
                        unique_lines.append(line)
                        seen.add(line)
                result = '\n'.join(unique_lines)
            elif operation == "sort_lines":
                lines = text.split('\n')
                reverse = options.get('reverse', False)
                result = '\n'.join(sorted(lines, reverse=reverse))
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
            
            return {
                "success": True,
                "original_text": text,
                "transformed_text": result,
                "operation": operation,
                "original_length": len(text),
                "new_length": len(result)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class TextSearchTool(BaseTool):
    """Tool for searching and replacing text."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="text_search",
            description="Search, find, and replace text using patterns and regular expressions",
            category=ToolCategory.UTILITY,
            version="1.0.0",
            author="System",
            permissions=[],
            examples=[
                'text_search(text="Hello world", pattern="world", action="find")',
                'text_search(text="Hello world", pattern="world", replacement="universe", action="replace")',
                'text_search(text="Test 123", pattern="\\\\d+", action="find", use_regex=true)'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="str",
                description="Text to search in",
                required=True
            ),
            ToolParameter(
                name="pattern",
                type="str",
                description="Pattern to search for",
                required=True
            ),
            ToolParameter(
                name="action",
                type="str",
                description="Action to perform",
                required=True,
                choices=["find", "replace", "split", "extract"]
            ),
            ToolParameter(
                name="replacement",
                type="str",
                description="Replacement text (for replace action)",
                required=False,
                default=""
            ),
            ToolParameter(
                name="use_regex",
                type="bool",
                description="Use regular expressions",
                required=False,
                default=False
            ),
            ToolParameter(
                name="case_sensitive",
                type="bool",
                description="Case sensitive search",
                required=False,
                default=True
            ),
            ToolParameter(
                name="max_results",
                type="int",
                description="Maximum number of results to return",
                required=False,
                default=100
            )
        ]
    
    async def _execute(
        self,
        text: str,
        pattern: str,
        action: str,
        replacement: str = "",
        use_regex: bool = False,
        case_sensitive: bool = True,
        max_results: int = 100
    ) -> Dict[str, Any]:
        """Search and manipulate text."""
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            
            if action == "find":
                if use_regex:
                    matches = re.finditer(pattern, text, flags)
                    results = []
                    for i, match in enumerate(matches):
                        if i >= max_results:
                            break
                        results.append({
                            "match": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                            "groups": match.groups()
                        })
                else:
                    results = []
                    search_text = text.lower() if not case_sensitive else text
                    search_pattern = pattern.lower() if not case_sensitive else pattern
                    
                    start = 0
                    while len(results) < max_results:
                        pos = search_text.find(search_pattern, start)
                        if pos == -1:
                            break
                        results.append({
                            "match": text[pos:pos + len(pattern)],
                            "start": pos,
                            "end": pos + len(pattern)
                        })
                        start = pos + 1
                
                return {
                    "action": "find",
                    "pattern": pattern,
                    "matches_found": len(results),
                    "matches": results
                }
            
            elif action == "replace":
                if use_regex:
                    result_text = re.sub(pattern, replacement, text, flags=flags)
                    count = len(re.findall(pattern, text, flags))
                else:
                    if case_sensitive:
                        result_text = text.replace(pattern, replacement)
                        count = text.count(pattern)
                    else:
                        # Case insensitive replace
                        import re
                        result_text = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)
                        count = len(re.findall(re.escape(pattern), text, re.IGNORECASE))
                
                return {
                    "action": "replace",
                    "pattern": pattern,
                    "replacement": replacement,
                    "replacements_made": count,
                    "original_text": text,
                    "result_text": result_text
                }
            
            elif action == "split":
                if use_regex:
                    parts = re.split(pattern, text, flags=flags)
                else:
                    parts = text.split(pattern)
                
                return {
                    "action": "split",
                    "pattern": pattern,
                    "parts_count": len(parts),
                    "parts": parts[:max_results]
                }
            
            elif action == "extract":
                if use_regex:
                    matches = re.findall(pattern, text, flags)
                    results = matches[:max_results]
                else:
                    # For non-regex, extract lines containing the pattern
                    lines = text.split('\n')
                    search_pattern = pattern.lower() if not case_sensitive else pattern
                    results = []
                    
                    for line in lines:
                        search_line = line.lower() if not case_sensitive else line
                        if search_pattern in search_line:
                            results.append(line)
                            if len(results) >= max_results:
                                break
                
                return {
                    "action": "extract",
                    "pattern": pattern,
                    "extracted_count": len(results),
                    "extracted": results
                }
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        
        except re.error as e:
            return {"success": False, "error": f"Regex error: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class TextGeneratorTool(BaseTool):
    """Tool for generating text content."""
    
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="text_generator",
            description="Generate various types of text content and templates",
            category=ToolCategory.PRODUCTIVITY,
            version="1.0.0",
            author="System",
            permissions=[],
            examples=[
                'text_generator(type="lorem", length=100)',
                'text_generator(type="password", length=16)',
                'text_generator(type="uuid")',
                'text_generator(type="template", template="Hello {name}!", data={"name": "World"})'
            ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="type",
                type="str",
                description="Type of text to generate",
                required=True,
                choices=["lorem", "password", "uuid", "random_string", "template", "list", "table"]
            ),
            ToolParameter(
                name="length",
                type="int",
                description="Length of generated text (words for lorem, characters for others)",
                required=False,
                default=50
            ),
            ToolParameter(
                name="template",
                type="str",
                description="Template string with placeholders (for template type)",
                required=False,
                default=""
            ),
            ToolParameter(
                name="data",
                type="dict",
                description="Data to fill template placeholders",
                required=False,
                default={}
            ),
            ToolParameter(
                name="options",
                type="dict",
                description="Additional options for generation",
                required=False,
                default={}
            )
        ]
    
    async def _execute(
        self,
        type: str,
        length: int = 50,
        template: str = "",
        data: Dict[str, Any] = None,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate text content."""
        if data is None:
            data = {}
        if options is None:
            options = {}
        
        try:
            if type == "lorem":
                lorem_words = [
                    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
                    "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
                    "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
                    "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
                    "consequat", "duis", "aute", "irure", "in", "reprehenderit", "voluptate",
                    "velit", "esse", "cillum", "fugiat", "nulla", "pariatur", "excepteur", "sint",
                    "occaecat", "cupidatat", "non", "proident", "sunt", "culpa", "qui", "officia",
                    "deserunt", "mollit", "anim", "id", "est", "laborum"
                ]
                
                import random
                words = []
                for _ in range(length):
                    words.append(random.choice(lorem_words))
                
                result = ' '.join(words).capitalize() + '.'
                
            elif type == "password":
                import random
                import string
                
                include_upper = options.get('include_upper', True)
                include_lower = options.get('include_lower', True)
                include_digits = options.get('include_digits', True)
                include_symbols = options.get('include_symbols', True)
                
                chars = ""
                if include_upper:
                    chars += string.ascii_uppercase
                if include_lower:
                    chars += string.ascii_lowercase
                if include_digits:
                    chars += string.digits
                if include_symbols:
                    chars += "!@#$%^&*()-_=+[]{}|;:,.<>?"
                
                if not chars:
                    chars = string.ascii_letters + string.digits
                
                result = ''.join(random.choice(chars) for _ in range(length))
                
            elif type == "uuid":
                import uuid
                uuid_version = options.get('version', 4)
                
                if uuid_version == 1:
                    result = str(uuid.uuid1())
                elif uuid_version == 4:
                    result = str(uuid.uuid4())
                else:
                    result = str(uuid.uuid4())
                    
            elif type == "random_string":
                import random
                import string
                
                chars = options.get('chars', string.ascii_letters + string.digits)
                result = ''.join(random.choice(chars) for _ in range(length))
                
            elif type == "template":
                if not template:
                    return {"success": False, "error": "Template string required"}
                
                try:
                    result = template.format(**data)
                except KeyError as e:
                    return {"success": False, "error": f"Missing template variable: {e}"}
                except Exception as e:
                    return {"success": False, "error": f"Template error: {e}"}
                    
            elif type == "list":
                items = options.get('items', ['Item 1', 'Item 2', 'Item 3'])
                list_type = options.get('list_type', 'bullet')  # bullet, numbered, checkbox
                
                if list_type == 'numbered':
                    result = '\n'.join(f"{i+1}. {item}" for i, item in enumerate(items))
                elif list_type == 'checkbox':
                    result = '\n'.join(f"- [ ] {item}" for item in items)
                else:  # bullet
                    result = '\n'.join(f"- {item}" for item in items)
                    
            elif type == "table":
                headers = options.get('headers', ['Column 1', 'Column 2', 'Column 3'])
                rows = options.get('rows', [['Row 1 Col 1', 'Row 1 Col 2', 'Row 1 Col 3']])
                
                # Simple text table
                col_widths = [max(len(str(header)), max(len(str(row[i])) if i < len(row) else 0 for row in rows)) 
                             for i, header in enumerate(headers)]
                
                # Header
                header_line = '| ' + ' | '.join(header.ljust(col_widths[i]) for i, header in enumerate(headers)) + ' |'
                separator = '|' + '|'.join('-' * (width + 2) for width in col_widths) + '|'
                
                # Rows
                row_lines = []
                for row in rows:
                    row_line = '| ' + ' | '.join(
                        str(row[i]).ljust(col_widths[i]) if i < len(row) else ''.ljust(col_widths[i])
                        for i in range(len(headers))
                    ) + ' |'
                    row_lines.append(row_line)
                
                result = '\n'.join([header_line, separator] + row_lines)
                
            else:
                return {"success": False, "error": f"Unknown generation type: {type}"}
            
            return {
                "success": True,
                "type": type,
                "generated_text": result,
                "length": len(result),
                "options_used": options
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}