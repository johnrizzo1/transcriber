# Query Command Feature Context

## Feature Overview

**Feature Name**: Query Command with Vector Memory  
**Status**: Planned - Implementation Ready  
**Priority**: High  
**Estimated Timeline**: 4 weeks  

## What This Feature Adds

### Core Functionality
- **Single Query Processing**: `poetry run transcriber query "What tools do you have access to?"`
- **Persistent Memory**: Remember information across CLI invocations using ChromaDB
- **Semantic Context**: Retrieve relevant past conversations using vector similarity
- **Personal Information Storage**: Store and recall user details like names, preferences

### Example Usage Flow
```bash
# Store personal info
poetry run transcriber query "My name is John Rizzo"

# Later retrieve it
poetry run transcriber query "What is my name?"
# Response: "Your name is John Rizzo."
```

## Technical Architecture

### Memory System Components
1. **ChromaDB Storage**: Local vector database for embeddings
2. **SentenceTransformers**: Local embedding generation (all-MiniLM-L6-v2)
3. **Memory Manager**: High-level interface for context retrieval
4. **Query Agent**: Specialized agent for single-query processing

### Integration Points
- **Existing Agent System**: Extends VoiceAgent with memory capabilities
- **Session Management**: Bridges with existing SQLite session storage
- **Configuration System**: New MemoryConfig section in settings
- **CLI Framework**: New commands alongside existing `chat` and `start`

## Implementation Plan Location

**Full Technical Specification**: [`docs/QUERY_COMMAND_IMPLEMENTATION_PLAN.md`](../docs/QUERY_COMMAND_IMPLEMENTATION_PLAN.md)

The plan includes:
- Detailed architecture diagrams
- Complete code implementations for all components
- Phase-by-phase development timeline
- Comprehensive testing strategy
- Performance requirements and optimization
- Risk mitigation strategies

## Key Design Decisions

### ChromaDB Choice
- **Local-First**: Maintains project's privacy-first approach
- **Persistent**: Survives application restarts
- **Efficient**: Optimized vector similarity search
- **Python Native**: Seamless integration with existing codebase

### Memory Strategy
- **Semantic Similarity**: Uses embeddings for context relevance
- **Temporal Ranking**: Recent memories weighted higher
- **Background Processing**: Non-blocking memory storage
- **Automatic Cleanup**: Configurable retention policies

### User Experience
- **Simple CLI**: Single command with intuitive options
- **Verbose Mode**: Optional memory context visibility
- **Privacy Controls**: Options to disable memory per query
- **Statistics**: Memory system status and management commands

## Development Phases

### Phase 1: Memory System Foundation (Week 1)
- Memory models and configuration
- Embedding service with fallbacks
- ChromaDB storage layer
- Basic unit tests

### Phase 2: Core Memory System (Week 2)
- Memory manager with context retrieval
- Background processing and cleanup
- Performance optimization
- Comprehensive testing

### Phase 3: Query Command (Week 3)
- Query agent implementation
- CLI command with all options
- Integration with existing systems
- Management commands

### Phase 4: Integration & Polish (Week 4)
- End-to-end testing
- Performance validation
- Documentation completion
- Bug fixes and refinements

## Success Metrics

### Functional Goals
- ✅ Single query processing with memory
- ✅ Cross-invocation persistence
- ✅ Personal information recall
- ✅ Semantic context retrieval

### Performance Targets
- Query processing: < 2 seconds total
- Memory retrieval: < 200ms
- Background storage: < 500ms
- Embedding generation: < 1 second

## Impact on Existing System

### Minimal Disruption
- **Additive Feature**: Doesn't modify existing functionality
- **Optional Memory**: Can be disabled via configuration
- **Separate Commands**: New `query` command alongside existing `chat`
- **Backward Compatible**: Existing workflows unchanged

### Enhanced Capabilities
- **Persistent Context**: Agent remembers across sessions
- **Personalization**: Tailored responses based on history
- **Improved UX**: Single-command interaction option
- **Analytics**: Memory usage statistics and management

## Next Steps

1. **Review and Approval**: Validate technical approach and requirements
2. **Environment Setup**: Add ChromaDB and sentence-transformers dependencies
3. **Phase 1 Implementation**: Begin with memory system foundation
4. **Iterative Development**: Follow 4-week phased approach
5. **Testing and Validation**: Comprehensive testing throughout development

## Related Documentation

- **Main Implementation Plan**: [`docs/QUERY_COMMAND_IMPLEMENTATION_PLAN.md`](../docs/QUERY_COMMAND_IMPLEMENTATION_PLAN.md)
- **Project Architecture**: [`memory-bank/systemPatterns.md`](systemPatterns.md)
- **Current Progress**: [`memory-bank/progress.md`](progress.md)
- **Technical Context**: [`memory-bank/techContext.md`](techContext.md)

This feature represents a significant enhancement to the transcriber's capabilities, adding persistent memory and single-query processing while maintaining the project's core principles of privacy, performance, and local processing.