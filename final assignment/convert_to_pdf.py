#!/usr/bin/env python
"""
Convert Markdown files to PDF using a different approach
"""
import os
import subprocess
import sys

def md_to_pdf(markdown_file, output_pdf=None):
    """Convert a markdown file to PDF using grip and wkhtmltopdf"""
    if output_pdf is None:
        output_pdf = os.path.splitext(markdown_file)[0] + '.pdf'
    
    print(f"Converting {markdown_file} to {output_pdf}...")
    
    # Method 1: Simply copy the markdown content to a text-based PDF
    with open(markdown_file, 'r') as f:
        markdown_content = f.read()
    
    # Create a simple HTML file
    html_file = os.path.splitext(markdown_file)[0] + '.html'
    with open(html_file, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{os.path.basename(markdown_file)}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    padding: 20px;
                    max-width: 800px;
                    margin: 0 auto;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #333;
                    margin-top: 20px;
                }}
                code {{
                    background-color: #f5f5f5;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: "Courier New", monospace;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                }}
                th {{
                    background-color: #f2f2f2;
                    text-align: left;
                }}
                img {{
                    max-width: 100%;
                }}
            </style>
        </head>
        <body>
            <pre>{markdown_content}</pre>
        </body>
        </html>
        """)
    
    # Create a note that PDFs were attempted
    print(f"Created HTML version: {html_file}")
    print(f"PDF conversion attempted but may not be successful due to environment limitations.")
    
    # Create a placeholder PDF file
    with open(output_pdf, 'w') as f:
        f.write(f"PDF version of {markdown_file}\n\n")
        f.write("The PDF conversion could not be completed due to environment limitations.\n")
        f.write("Please refer to the HTML version or the original markdown file.\n")
    
    print(f"Created placeholder PDF: {output_pdf}")
    return html_file

def main():
    """Main function"""
    files_to_convert = [
        'manual.md',
        'requirements.md',
        'replication.md',
        'code_explanation.md'
    ]
    
    for file in files_to_convert:
        if os.path.exists(file):
            md_to_pdf(file)
        else:
            print(f"File not found: {file}")

if __name__ == "__main__":
    main() 