# Stack Overflow Auto Tagging System

A machine learning-powered web application that automatically suggests relevant tags for technical questions, optimized for Stack Overflow-style queries.

## Project Overview
This system analyzes both the title and body text of technical questions to predict appropriate tags using two trained classifiers. Key components:

- **Dual-classifier architecture** combining predictions from title and body analysis
- **Custom text preprocessing pipeline** with HTML cleaning and NLP techniques
- **Production-ready web interface** built with Flask
- **Pre-trained models** using TF-IDF vectorization and Linear SVM

## Key Features
- **Multi-source prediction**: Combines results from separate title and body classifiers
- **Text normalization**:
  - HTML tag removal
  - Special character handling
  - Stemming and stopword removal
- **Threshold-based tag selection**: Only shows tags with prediction confidence ≥50%
- **Real-time processing**: Instant predictions through web interface
