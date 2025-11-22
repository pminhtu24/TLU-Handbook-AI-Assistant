from langchain_core.documents import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                TableStructureOptions)
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
import os, tempfile

from torch import device

OUTPUT_DIR = "/home/tupham/Documents/Development/Student_Handbook/output"

def load_data_from_upload(file_object) -> tuple:
    """
    Load data from uploaded file (Streamlit UploadedFile) & convert to Markdown
    
    Returns:
        (markdown_text, doc_name): Tuple 
    """
    
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,   
        table_structure_options=TableStructureOptions(do_cell_matching=True),
        accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    )
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    doc_name = getattr(file_object, 'name', 'unknown_document')
    if doc_name.endswith('.pdf'):
        doc_name = doc_name[:-4]
    
    # File temp
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_object.read())
            tmp_path = tmp_file.name
        
        result = converter.convert(tmp_path)
        os.unlink(tmp_path)

    except Exception as e:
        raise Exception(f"Error when convert PDF: {e}")
    
    
    doc = result.document
    if not doc:
        raise Exception("Docling cant process this file !")
    
    # Export to Markdown
    markdown_text = doc.export_to_markdown()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_DIR, f"{doc_name}.md")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Markdown saved: {output_file_path}")
    return markdown_text, doc_name



def load_data_from_local(file_path: str) -> tuple:
    """
    Args:
        file_path (str): Full path to file
    
    Returns:
        (markdown_text, doc_name): Tuple     
    
    """
    supported_format = ('.pdf')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith(supported_format):
        raise ValueError("Only supported format PDF, DOCX !")
    
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(do_cell_matching=True),
        accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    doc_name = os.path.basename(file_path)
    if doc_name.endswith('.pdf'):
        doc_name = doc_name[-4]

    try:
        result = converter.convert(file_path)
    except Exception as e:
        raise Exception(f"Error when convert file: {e}")
    
    doc = result.document
    if not doc:
        raise Exception("Docling can not process this file !")
    
    # Export to markdown
    markdown_text = doc.export_to_markdown()
    return markdown_text, doc_name



def load_data_from_directory(directory_path: str) -> list[tuple]:
    """
    Args:
        directory_path (str): Path to the folder containing the data files
        
    Returns:
        list: list of tuple  [(markdown_text, doc_name), ..]

    """

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"{directory_path} is not a directory !")
    

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise ValueError(f"None of PDF files found in: {directory_path} !")

    print(f"Got {len(pdf_files)} file PDF in folder")

    results = []
    failed_files = []

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        try:
            markdown_text, doc_name = load_data_from_local(file_path)
            results.append((markdown_text, doc_name))
            print(f"Loaded {pdf_file}")
        except Exception as e:
            failed_files.append((pdf_file, str(e)))
            print(f"Error when process {pdf_file}: {e}")
    
    if failed_files:
        print(f"Got {len(failed_files)} error files")
    
    if not results:
        raise Exception("Cannot process any PDF files in the folder")
    
    print(f"\n A total of {len(results)}/{len(pdf_files)} successfully processed file")
    return results


def create_documents(markdown_text, doc_name):
    doc_metadata = {
        'source': doc_name,
        'content_type': 'text/markdown',
        'title': doc_name,
        'language': 'vi',
    }

    document = Document(page_content=markdown_text, metadata=doc_metadata)
    print(f"Created Document for: {doc_name}")
    return document