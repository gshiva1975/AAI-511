#!/bin/bash

# Check the data directory structure
echo "=== Checking data directory structure ==="
DATA_DIR="/Users/gshiva/AA-511/example/mcp-server-example/mcp-server/sheetmusic/test_image/mutopia_pdfs"

echo "Directory exists check:"
ls -la "$DATA_DIR" 2>/dev/null || echo "Directory does not exist or is not accessible"

echo -e "\n=== Checking subdirectories (should be composer names) ==="
find "$DATA_DIR" -maxdepth 1 -type d ! -path "$DATA_DIR" 2>/dev/null

echo -e "\n=== Checking for PDF files ==="
find "$DATA_DIR" -name "*.pdf" | head -10

echo -e "\n=== Total PDF count ==="
find "$DATA_DIR" -name "*.pdf" | wc -l

echo -e "\n=== Directory structure (first 20 items) ==="
find "$DATA_DIR" -type f | head -20
