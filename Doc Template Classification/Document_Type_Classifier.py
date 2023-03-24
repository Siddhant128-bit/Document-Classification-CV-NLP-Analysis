# import streamlit as st
# import pdfminer
# from pdfminer.high_level import extract_pages
# def displayPDF(uploaded_file):
#     if uploaded_file is not None:
#         for page_layout in extract_pages(uploaded_file):
#             for element in page_layout:
#                 st.markdown(element)    

# with st.sidebar:
#     inference_file=st.sidebar.file_uploader('Upload Files: ')

# displayPDF(inference_file)

# #uploaded_file = st.file_uploader("Choose a file")


import base64
import tempfile
import streamlit as st
from pathlib import Path
from utilities import *
import shutil

def show_pdf(file_path:str):
    """Show the PDF in Streamlit
    That returns as html component

    Parameters
    ----------
    file_path : [str]
        Uploaded PDF file path
    """

    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

def update_inference_testing_pdf(file,on_off_flag):

    if on_off_flag==1:
        try:
            os.mkdir('Temp_Files')
        except:
            pass

        with open('Temp_Files/temp_pdf.pdf', 'wb') as f: 
            f.write(file.getvalue())
    else: 
        shutil.rmtree('Temp_Files')

    

def main():
    """Streamlit application
    """

    st.title("Document Classifier Glean")
    st.write('Upload Your File on the sidebar and This app will provide you with the Document Type Classified for your usecase')
    with st.sidebar:
        uploaded_file=st.sidebar.file_uploader('Upload Files: ',type=['pdf','.png','.jpg'])
    
    if uploaded_file is not None:
        update_inference_testing_pdf(uploaded_file,1)
        # Make temp file path from uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            st.markdown("## Original PDF file")
            fp = Path(tmp_file.name)
            fp.write_bytes(uploaded_file.getvalue())
            st.write(show_pdf(tmp_file.name))
            output=inference_testing_model('Efficient_net_custom.h5','output_map.json','Temp_Files/temp_pdf.pdf')
            update_inference_testing_pdf('',0)
            
        if output is not None: 
            st.write(output)


if __name__ == "__main__":
    main()