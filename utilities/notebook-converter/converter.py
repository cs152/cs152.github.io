import streamlit as st
import nbformat
from nbconvert import WebPDFExporter

pagebreak = r'''<div style="page-break-after: always; visibility: hidden"> 
\pagebreak 
</div>'''

st.title('CS152 Notebook Converter')
uploaded_file = st.file_uploader("Choose an ipynb file to convert to PDF for submission")
if uploaded_file is not None:
    # To read file as bytes:
    with st.spinner('Converting...'):
        nb = nbformat.read(uploaded_file, 4)
        cells = []
        for question in range(20):
            qstr = f'Q{question}'
            if question > 0:
                cells.append(nbformat.v4.new_markdown_cell(source='# ' + qstr))
            else:
                cells.append(nbformat.v4.new_markdown_cell(source='CS152 Submission'))

            for cell in nb.cells:
                checksource = cell.source.strip().replace(' ', '').find('#!' + qstr) == 0
                if ('tags' in cell.metadata and qstr in cell.metadata.tags) or checksource:
                    if checksource:
                        cell.source = '\n'.join(cell.source.splitlines()[1:])
                    cells.append(cell)
                    
            cells.append(nbformat.v4.new_markdown_cell(source=pagebreak))

        output = nbformat.v4.new_notebook()
        output.cells = cells
        (pdf, _) = WebPDFExporter().from_notebook_node(output)
        st.download_button('Download PDF', pdf, file_name='submission.pdf') 

