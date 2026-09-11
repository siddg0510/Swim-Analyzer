#!/usr/bin/env python3
"""Convenience runner script for Swim Analyzer.

Usage:
  python run.py          Launch desktop GUI
  python run.py web      Launch web server (http://localhost:8000)
"""
import sys
import main

if __name__ == "__main__":
    sys.exit(main.main())
