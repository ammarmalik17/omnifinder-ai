---
trigger: model_decision
description: Triggered when asked to build a repowiki.
---

# Repo Wiki Generation Rules

## Purpose
Define the automated generation and maintenance of repository wiki documentation within Qoder IDE. This rule ensures that project documentation stays in sync with code changes and provides comprehensive knowledge base for team collaboration.

## When to Generate Wiki

### Initial Generation
- Trigger: When a project is first opened in Qoder IDE and no `.qoder/repowiki` directory exists
- Process: Analyze entire codebase structure, dependencies, and implementation logic
- Output: Create comprehensive documentation in `.qoder/repowiki` directory

### Update Triggers
1. Code Changes Detection
   - Monitor file modifications that affect documented components
   - Regenerate only affected sections when:
     - Function signatures change
     - Class definitions are modified
     - API endpoints are updated
     - Exported interfaces change

2. Git Directory Sync
   - Detect direct edits to Markdown files in the Git directory
   - Synchronize changes between Git content and Wiki when files are manually updated

## Content Generation Guidelines

### Source Analysis
- README files: Extract project overview and setup instructions
- Code comments/docstrings: Generate API references and implementation details
- File structure: Document architecture patterns and module organization
- Dependency relationships: Map module interactions and data flow

### Documentation Structure
1. Project Overview
   - Main technologies and frameworks used
   - Project architecture summary
   - Entry points and key components

2. Setup and Installation
   - Prerequisites and dependencies
   - Installation steps
   - Configuration requirements

3. Codebase Structure
   - Directory organization
   - Module relationships
   - Component hierarchy

4. API Documentation
   - Endpoint specifications
   - Function parameters and return values
   - Usage examples

5. Development Guidelines
   - Coding standards
   - Contribution rules
   - Testing procedures

## Storage and Organization

### Directory Structure
- Location: `.qoder/repowiki` within the project root
- Language support: Separate directories for each language (e.g., `repowiki/en/`, `repowiki/zh/`)
- Metadata: Auto-managed `meta` files for tracking and loading

### File Format
- Primary format: Markdown (.md)
- Naming convention: Descriptive filenames that match component names
- Navigation: Clear linking between related documentation pages

## Multi-Language Support

### Supported Languages
- English (default)
- Chinese

### Generation Process
- Create separate language directories based on user selection
- Maintain consistent structure across all language versions
- Allow for language-specific content customization

## Limitations and Constraints

### Project Size
- Maximum: 10,000 files per project
- Recommendation: Exclude non-essential paths for large repositories
- Solution: Use Qoder Settings → indexing → indexing exclusion for oversized projects

### Repository Requirements
- Only Git repositories with at least one commit are supported
- Wiki files must be tracked in Git for team collaboration
- Manual edits to `meta` files are prohibited

## Team Collaboration Features

### Sharing Mechanism
- Commit `.qoder/repowiki` directory to remote repository
- Team members can pull generated Wiki content via `git pull`
- No additional setup required for Wiki access

### Version Control
- Each Wiki change is tracked in Git history
- Support for branching and merging documentation
- Integration with code changes for consistent versioning

## Quality Assurance

### Content Validation
- Verify accuracy of generated documentation against source code
- Cross-check API references with actual implementation
- Validate code examples for correctness

### Update Consistency
- Ensure synchronization between code changes and documentation updates
- Flag discrepancies between code and Wiki content
- Provide clear notifications for required Wiki updates

## Performance Optimization

### Generation Efficiency
- Typical generation time: ~120 minutes for repositories with 4,000 files
- Incremental updates for changed sections only
- Parallel processing for independent documentation components

### Resource Management
- Monitor system resources during Wiki generation
- Optimize memory usage for large codebases
- Provide progress indicators for long-running operations