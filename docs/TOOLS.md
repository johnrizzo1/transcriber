# Tools Documentation

Complete reference for all 15 built-in tools available in the AI Voice Agent.

## Table of Contents

1. [Overview](#overview)
2. [System Tools](#system-tools)
3. [Utility Tools](#utility-tools)
4. [Information Tools](#information-tools)
5. [Productivity Tools](#productivity-tools)
6. [Tool Usage Examples](#tool-usage-examples)
7. [Tool Development](#tool-development)

## Overview

The AI Voice Agent includes **15 built-in tools** across **4 categories** that enable the AI to perform various tasks beyond conversation. Tools are automatically invoked when the AI determines they're needed to fulfill your request.

### Tool Categories

| Category | Count | Description |
|----------|-------|-------------|
| **System** | 5 tools | File operations, process management, system information |
| **Utility** | 4 tools | Calculations, text processing, data manipulation |
| **Information** | 3 tools | Web search, documentation lookup, unit conversion |
| **Productivity** | 3 tools | Note-taking, task management, timers |

### Tool Permissions

Tools operate with different permission levels:

- 🟢 **READ** - Can read files and data
- 🟡 **WRITE** - Can modify files and data  
- 🔴 **EXECUTE** - Can run programs and commands
- 🔵 **NETWORK** - Can access network resources
- 🟣 **SYSTEM** - Can modify system settings

## System Tools

### 1. File Read Tool

**Purpose**: Read and display file contents

**Permissions**: 🟢 READ

**Usage Examples**:
- "Read the contents of README.md"
- "Show me what's in the config.py file"
- "Can you read the log file and tell me what happened?"

**Parameters**:
- `path` (required): File path to read
- `encoding` (optional): File encoding (default: utf-8)
- `max_lines` (optional): Maximum lines to read (default: 1000)

**Example Output**:
```
File: README.md (1,247 words, 156 lines)

# AI Voice Agent - Transcriber

A local, real-time voice interface for interacting with an AI agent...
```

### 2. File Write Tool

**Purpose**: Create or modify files

**Permissions**: 🟡 WRITE

**Usage Examples**:
- "Create a new file called notes.txt with my meeting notes"
- "Write a Python script that prints hello world"
- "Save this configuration to config.yaml"

**Parameters**:
- `path` (required): File path to write
- `content` (required): Content to write
- `mode` (optional): Write mode (write, append) (default: write)
- `encoding` (optional): File encoding (default: utf-8)

**Example Output**:
```
✅ File created successfully: notes.txt (245 bytes)
```

### 3. File List Tool

**Purpose**: List files and directories

**Permissions**: 🟢 READ

**Usage Examples**:
- "List all files in the current directory"
- "Show me all Python files in the src folder"
- "What files are in my Documents folder?"

**Parameters**:
- `path` (optional): Directory path (default: current directory)
- `pattern` (optional): File pattern filter (e.g., "*.py")
- `recursive` (optional): Include subdirectories (default: false)
- `show_hidden` (optional): Show hidden files (default: false)

**Example Output**:
```
Directory: /Users/user/project (12 items)

📁 docs/
📁 src/
📁 tests/
📄 README.md (2.1 KB)
📄 pyproject.toml (1.8 KB)
📄 .gitignore (456 bytes)
```

### 4. File Delete Tool

**Purpose**: Delete files and directories

**Permissions**: 🟡 WRITE

**Usage Examples**:
- "Delete the temporary file temp.log"
- "Remove the old backup folder"
- "Can you delete all .pyc files?"

**Parameters**:
- `path` (required): File or directory path to delete
- `recursive` (optional): Delete directories recursively (default: false)
- `confirm` (optional): Require confirmation (default: true)

**Example Output**:
```
⚠️  Are you sure you want to delete 'temp.log'? (y/N): y
✅ File deleted successfully: temp.log
```

### 5. File Copy Tool

**Purpose**: Copy files and directories

**Permissions**: 🟡 WRITE

**Usage Examples**:
- "Copy README.md to README_backup.md"
- "Make a backup of the config folder"
- "Copy all Python files to the backup directory"

**Parameters**:
- `source` (required): Source file or directory path
- `destination` (required): Destination path
- `overwrite` (optional): Overwrite existing files (default: false)
- `recursive` (optional): Copy directories recursively (default: true)

**Example Output**:
```
✅ Copied successfully: README.md → README_backup.md (2.1 KB)
```

## Utility Tools

### 6. Calculator Tool

**Purpose**: Perform mathematical calculations

**Permissions**: None

**Usage Examples**:
- "What's 15% of 250?"
- "Calculate the square root of 144"
- "What's 2 to the power of 8?"

**Parameters**:
- `expression` (required): Mathematical expression to evaluate
- `precision` (optional): Decimal places (default: 2)

**Supported Operations**:
- Basic arithmetic: `+`, `-`, `*`, `/`, `%`
- Powers: `**` or `^`
- Functions: `sqrt()`, `sin()`, `cos()`, `tan()`, `log()`, `ln()`
- Constants: `pi`, `e`

**Example Output**:
```
Expression: 15% of 250
Calculation: 250 * 0.15
Result: 37.50
```

### 7. Advanced Calculator Tool

**Purpose**: Complex mathematical operations and equation solving

**Permissions**: None

**Usage Examples**:
- "Solve the quadratic equation x² + 5x + 6 = 0"
- "Calculate the derivative of x³ + 2x² - 5x + 1"
- "What's the integral of sin(x) from 0 to π?"

**Parameters**:
- `expression` (required): Mathematical expression or equation
- `operation` (optional): Type of operation (evaluate, solve, derivative, integral)
- `variable` (optional): Variable name (default: x)

**Example Output**:
```
Equation: x² + 5x + 6 = 0
Solutions: x = -2, x = -3
Verification: (-2)² + 5(-2) + 6 = 4 - 10 + 6 = 0 ✓
```

### 8. Text Analysis Tool

**Purpose**: Analyze text content and statistics

**Permissions**: None

**Usage Examples**:
- "Analyze this text and count the words"
- "What's the reading level of this document?"
- "Check the sentiment of this review"

**Parameters**:
- `text` (required): Text to analyze
- `analysis_type` (optional): Type of analysis (stats, sentiment, readability)

**Analysis Types**:
- **Stats**: Word count, character count, sentences, paragraphs
- **Sentiment**: Positive, negative, neutral sentiment analysis
- **Readability**: Reading level, complexity scores

**Example Output**:
```
Text Analysis Results:
📊 Statistics:
  - Words: 247
  - Characters: 1,456
  - Sentences: 18
  - Paragraphs: 4
  - Average words per sentence: 13.7

😊 Sentiment: Positive (0.72)
📚 Reading Level: Grade 8-9 (Flesch-Kincaid: 8.4)
```

### 9. Text Transform Tool

**Purpose**: Transform and manipulate text

**Permissions**: None

**Usage Examples**:
- "Convert this text to uppercase"
- "Remove all punctuation from this sentence"
- "Replace all instances of 'old' with 'new'"

**Parameters**:
- `text` (required): Text to transform
- `operation` (required): Transformation operation
- `target` (optional): Target string for replace operations
- `replacement` (optional): Replacement string

**Operations**:
- `uppercase`, `lowercase`, `title_case`, `sentence_case`
- `remove_punctuation`, `remove_whitespace`, `remove_numbers`
- `replace`, `reverse`, `sort_words`

**Example Output**:
```
Original: "Hello, World! How are you today?"
Operation: uppercase
Result: "HELLO, WORLD! HOW ARE YOU TODAY?"
```

## Information Tools

### 10. Web Search Tool

**Purpose**: Search the web for information (local processing)

**Permissions**: 🔵 NETWORK

**Usage Examples**:
- "Search for information about Python asyncio"
- "Find recent news about artificial intelligence"
- "Look up the weather in San Francisco"

**Parameters**:
- `query` (required): Search query
- `num_results` (optional): Number of results (default: 5)
- `safe_search` (optional): Enable safe search (default: true)

**Example Output**:
```
Search Results for "Python asyncio":

1. 📄 Python asyncio Documentation
   https://docs.python.org/3/library/asyncio.html
   Official Python documentation for asyncio library...

2. 📄 Real Python - Async IO in Python
   https://realpython.com/async-io-python/
   A complete guide to asynchronous programming...

Found 5 results in 0.8 seconds
```

### 11. Documentation Lookup Tool

**Purpose**: Search documentation and help files

**Permissions**: 🟢 READ

**Usage Examples**:
- "Find help for the 'ls' command"
- "Show me Python documentation for the 'asyncio' module"
- "Look up usage examples for 'git commit'"

**Parameters**:
- `topic` (required): Topic or command to look up
- `source` (optional): Documentation source (man, python, git, etc.)
- `section` (optional): Specific section to search

**Example Output**:
```
Documentation: ls command

NAME
     ls -- list directory contents

SYNOPSIS
     ls [-ABCFGHLOPRSTUW@abcdefghiklmnopqrstuwx1] [file ...]

DESCRIPTION
     For each operand that names a file of a type other than directory,
     ls displays its name as well as any requested, associated information.
```

### 12. Unit Conversion Tool

**Purpose**: Convert between different units of measurement

**Permissions**: None

**Usage Examples**:
- "Convert 100 fahrenheit to celsius"
- "How many kilometers is 50 miles?"
- "Convert 2.5 hours to minutes"

**Parameters**:
- `value` (required): Numeric value to convert
- `from_unit` (required): Source unit
- `to_unit` (required): Target unit
- `category` (optional): Unit category (temperature, distance, time, etc.)

**Supported Categories**:
- **Temperature**: Celsius, Fahrenheit, Kelvin
- **Distance**: Meters, kilometers, miles, feet, inches
- **Weight**: Grams, kilograms, pounds, ounces
- **Time**: Seconds, minutes, hours, days
- **Volume**: Liters, gallons, cups, milliliters

**Example Output**:
```
Conversion: 100°F to °C
Formula: (°F - 32) × 5/9
Result: 37.78°C
```

## Productivity Tools

### 13. Note Taking Tool

**Purpose**: Create and manage notes

**Permissions**: 🟡 WRITE

**Usage Examples**:
- "Create a note titled 'Meeting Notes' with today's agenda"
- "Add a note about the new project requirements"
- "Show me all my notes from this week"

**Parameters**:
- `title` (required): Note title
- `content` (required): Note content
- `tags` (optional): Tags for organization
- `category` (optional): Note category

**Example Output**:
```
📝 Note Created: "Meeting Notes"
📅 Date: 2024-01-15 14:30
🏷️  Tags: work, meeting, project
📄 Content: 247 words

Note saved to: ~/.transcriber/notes/meeting-notes-20240115.md
```

### 14. Task Management Tool

**Purpose**: Create and track TODO items

**Permissions**: 🟡 WRITE

**Usage Examples**:
- "Add a task to review the documentation"
- "Mark the 'setup environment' task as completed"
- "Show me all pending tasks"

**Parameters**:
- `action` (required): Action (add, complete, list, delete)
- `task` (optional): Task description
- `priority` (optional): Priority level (low, medium, high)
- `due_date` (optional): Due date

**Example Output**:
```
✅ Task Management

📋 Current Tasks (3 active):
1. [HIGH] Review documentation (Due: Today)
2. [MED]  Setup CI/CD pipeline (Due: Jan 20)
3. [LOW]  Update README (No due date)

✅ Completed Today (2):
- Setup development environment
- Install dependencies
```

### 15. Timer/Reminder Tool

**Purpose**: Set timers and reminders

**Permissions**: None

**Usage Examples**:
- "Set a timer for 25 minutes for a Pomodoro session"
- "Remind me in 1 hour to check the server logs"
- "Show me all active timers"

**Parameters**:
- `action` (required): Action (set, list, cancel)
- `duration` (optional): Timer duration
- `message` (optional): Reminder message
- `type` (optional): Timer type (timer, reminder, alarm)

**Example Output**:
```
⏰ Timer Set: Pomodoro Session
⏱️  Duration: 25 minutes
🔔 Notification: "Time for a break!"
🆔 Timer ID: pomo-001

Active Timers (2):
1. Pomodoro Session - 23:45 remaining
2. Server Check Reminder - 58:30 remaining
```

## Tool Usage Examples

### Natural Language Requests

The AI automatically selects and uses appropriate tools based on your natural language requests:

**File Operations**:
```
You: "Can you read my config file and tell me what database it's using?"
AI: I'll read your config file and check the database settings.
[Uses: file_read tool]
AI: Based on your config file, you're using PostgreSQL as your database...
```

**Calculations**:
```
You: "If I have a 15% discount on a $250 item, how much will I save?"
AI: Let me calculate that discount for you.
[Uses: calculator tool]
AI: With a 15% discount on $250, you'll save $37.50, making the final price $212.50.
```

**Text Processing**:
```
You: "Take this paragraph and count how many words it has, then convert it to title case"
AI: I'll analyze the text and transform it for you.
[Uses: text_analysis tool, then text_transform tool]
AI: Your paragraph has 47 words. Here's the title case version: "This Is Your..."
```

### Complex Multi-Tool Operations

The AI can chain multiple tools together:

```
You: "Read the README file, count the words, and create a summary note"
AI: I'll read the README, analyze it, and create a summary note for you.
[Uses: file_read → text_analysis → note_taking tools]
AI: I've read your README (1,247 words), analyzed it, and created a summary note...
```

## Tool Development

### Creating Custom Tools

See [`TOOL_DEVELOPMENT.md`](TOOL_DEVELOPMENT.md) for detailed instructions on creating custom tools.

### Tool Architecture

All tools inherit from the `BaseTool` class and implement:

```python
class CustomTool(BaseTool):
    def _define_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="custom_tool",
            description="Description of what the tool does",
            category=ToolCategory.UTILITY,
            permissions=[ToolPermission.READ]
        )
    
    def _define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="str",
                description="Input parameter",
                required=True
            )
        ]
    
    async def _execute(self, **kwargs) -> Any:
        # Tool implementation
        return result
```

### Tool Registry

Tools are automatically discovered and registered:

```bash
# List all available tools
transcriber list-tools

# Get detailed tool information
transcriber list-tools --detailed

# Search for specific tools
transcriber list-tools --search "file"
```

---

This documentation covers all 15 built-in tools. For more information about using tools in conversations, see the [User Guide](USER_GUIDE.md).