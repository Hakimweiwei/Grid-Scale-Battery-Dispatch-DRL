import nbformat
import os

def convert_py_to_ipynb(py_script_path, ipynb_output_path):
    print(f"Converting {py_script_path} to {ipynb_output_path}...")
    nb = nbformat.v4.new_notebook()
    
    with open(py_script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Simple parser: split by '# %%' or block comments, but since we didn't use cell markers,
    # we'll extract the first few lines of comments as Markdown, and the rest as Code.
    lines = code.split('\n')
    markdown_lines = []
    code_lines = []
    
    in_markdown = True
    for line in lines:
        if in_markdown and line.startswith('#'):
            markdown_lines.append(line.lstrip('# ').strip())
        elif in_markdown and not line.strip():
            continue
        else:
            in_markdown = False
            code_lines.append(line)
            
    if markdown_lines:
        nb.cells.append(nbformat.v4.new_markdown_cell('\n'.join(markdown_lines)))
        
    if code_lines:
        nb.cells.append(nbformat.v4.new_code_cell('\n'.join(code_lines)))
        
    with open(ipynb_output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully generated {ipynb_output_path}")

if __name__ == '__main__':
    convert_py_to_ipynb('notebooks/01_eda_script.py', 'notebooks/01_eda_price_volatility.ipynb')
    convert_py_to_ipynb('notebooks/02_eda_script.py', 'notebooks/02_eda_duck_curve_and_fcas.ipynb')
